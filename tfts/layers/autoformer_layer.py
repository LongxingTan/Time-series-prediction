"""Layer for :py:class:`~tfts.models.autoformer`"""

from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling1D, Conv1D, Dense, Dropout


class MovingAvg(tf.keras.layers.Layer):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Moving average kernel size must be an odd number")
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super().build(input_shape)
        self.avg = AveragePooling1D(pool_size=self.kernel_size, strides=self.stride, padding="valid")

    def call(self, inputs):
        """
        Perform moving average for sequence

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor.
        """
        front = tf.tile(inputs[:, :1, :], [1, (self.kernel_size - 1) // 2, 1])
        end = tf.tile(inputs[:, -1:, :], [1, (self.kernel_size - 1) // 2, 1])
        x = tf.concat([front, inputs, end], axis=1)
        x = self.avg(x)
        return x


class SeriesDecomp(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int, name=None) -> None:
        super().__init__(name=name)
        self.kernel_size = kernel_size

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super().build(input_shape)
        self.moving_avg = MovingAvg(self.kernel_size, stride=1)

    def call(self, x: tf.Tensor):
        """
        Perform time-series decomposition on the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            A 3D tensor with shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple of two 3D tensors:
            - The residual tensor, shape (batch_size, sequence_length, input_dim).
            - The moving average tensor, which is a smoothed version of the input tensor.
        """
        moving_mean = self.moving_avg(x)
        trend = x - moving_mean
        return trend, moving_mean

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AutoCorrelation(tf.keras.layers.Layer):
    """Self-Attention layer that computes time-delayed autocorrelation between queries and keys.

    This layer implements a novel attention mechanism that uses Fast Fourier Transform (FFT)
    to compute autocorrelation between queries and keys in the frequency domain,
    which captures temporal dependencies more efficiently than traditional attention.

    Parameters
    ----------
    d_model : int
        The dimension of the model's hidden states.
    num_attention_heads : int
        Number of attention heads to use.
    attention_probs_dropout_prob : float, optional
        Dropout probability for attention probabilities, by default 0.0.
    """

    def __init__(self, d_model: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.0) -> None:
        super().__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError(f"Hidden size {d_model} must be divisible by the number of heads {num_attention_heads}.")
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_size = d_model // num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the layer, creating the trainable weights.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            The shape of the input tensor.
        """
        self.wq = Dense(self.d_model, name="q")
        self.wk = Dense(self.d_model, name="k")
        self.wv = Dense(self.d_model, name="v")
        self.drop = Dropout(self.attention_probs_dropout_prob)
        self.dense = Dense(self.d_model, name="project")
        super().build(input_shape)

    def time_delay_agg(self, q, k, v):
        """Compute time-delayed autocorrelation between queries and keys.

        Parameters
        ----------
        q : Tensor of shape (batch_size, num_attention_heads, time_steps, hidden_size)
            Queries.
        k : Tensor of shape (batch_size, num_attention_heads, time_steps, hidden_size)
            Keys.
        v : Tensor of shape (batch_size, num_attention_heads, time_steps, hidden_size)
            Values.

        Returns
        -------
        Tensor of shape (batch_size, num_attention_heads, hidden_size, time_steps)
            Time-delayed autocorrelation between queries and keys.
        """
        batch_size = tf.shape(q)[0]
        time_steps = tf.shape(q)[2]

        # Transform to frequency domain using FFT
        q_fft = tf.signal.rfft(tf.transpose(q, perm=[0, 1, 3, 2]))
        k_fft = tf.signal.rfft(tf.transpose(k, perm=[0, 1, 3, 2]))

        # Cross-correlation in frequency domain (multiplication with complex conjugate)
        S_qk = q_fft * tf.math.conj(k_fft)

        # Transform back to time domain
        R_qk = tf.signal.irfft(S_qk)

        # Create indices for the time steps
        init_index = tf.reshape(tf.range(time_steps), (1, 1, 1, -1))
        init_index = tf.tile(init_index, [batch_size, self.num_attention_heads, self.hidden_size, 1])

        # Use a fixed number of top correlations instead of dynamic calculation
        # This avoids the issue with symbolic tensors in range()
        top_k = 8  # A reasonable default value based on typical sequence lengths

        # Get top-k values and their indices
        weights, indices = tf.math.top_k(R_qk, k=top_k)

        # Apply softmax to get attention weights
        tmp_corr = tf.nn.softmax(weights, axis=-1)

        # Prepare values tensor with concatenated repetition for circular handling
        tmp_values = tf.tile(tf.transpose(q, perm=[0, 1, 3, 2]), [1, 1, 1, 2])
        delays_agg = tf.zeros_like(tf.transpose(q, perm=[0, 1, 3, 2]))

        # Aggregate values based on top-k correlations using tf.map_fn instead of Python loop
        def process_correlation(i):
            pattern = tf.gather(tmp_values, init_index + tf.expand_dims(indices[..., i], -1), axis=-1, batch_dims=-1)
            return pattern * tf.expand_dims(tmp_corr[..., i], axis=-1)

        # Generate a list of indices for our top_k
        indices_list = tf.range(top_k)

        # Apply the function to each index and sum the results
        correlation_patterns = tf.map_fn(
            process_correlation, indices_list, fn_output_signature=tf.transpose(q, perm=[0, 1, 3, 2]).dtype
        )
        delays_agg = tf.reduce_sum(correlation_patterns, axis=0)

        return delays_agg

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Parameters
        ----------
        x : Tensor
            Input tensor to split.
        batch_size : int
            Batch size.

        Returns
        -------
        Tensor
            Reshaped tensor with shape (batch_size, num_attention_heads, timesteps, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, dynamic=True):
        """Process inputs through the autocorrelation mechanism.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (batch_size, timesteps, d_model).
        k : Tensor
            Key tensor of shape (batch_size, timesteps, d_model).
        v : Tensor
            Value tensor of shape (batch_size, timesteps, d_model).
        dynamic : bool, optional
            Not used in the current implementation, by default True.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, timesteps, d_model).
        """
        batch_size = tf.shape(q)[0]

        # Apply linear projections
        q = self.drop(self.wq(q))
        k = self.drop(self.wk(k))
        v = self.drop(self.wv(v))

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Get sequence lengths
        L = tf.shape(q)[2]
        S = tf.shape(v)[2]

        # Handle sequence length differences using tf.cond instead of Python conditionals
        def pad_kv():
            zeros = tf.zeros_like(q[:, :, : (L - S), :])
            padded_v = tf.concat([v, zeros], axis=2)
            padded_k = tf.concat([k, zeros], axis=2)
            return padded_v, padded_k

        def trim_kv():
            return v[:, :, :L, :], k[:, :, :L, :]

        # Use tf.cond for graph-compatible conditional operations
        v_adjusted, k_adjusted = tf.cond(tf.greater(L, S), true_fn=pad_kv, false_fn=trim_kv)

        # Compute time-delayed autocorrelation
        delays_agg = self.time_delay_agg(q, k_adjusted, v_adjusted)
        delays_agg = tf.transpose(delays_agg, [0, 3, 1, 2])

        # Reshape and project to output dimension
        concat_delays_agg = tf.reshape(delays_agg, (batch_size, -1, self.d_model))
        output = self.dense(concat_delays_agg)

        return output

    def compute_output_spec(self, inputs_spec):
        """Compute the output tensor spec from the input spec.

        This is needed for TensorFlow 2.x keras model API.

        Parameters
        ----------
        inputs_spec : tf.TensorSpec
            Input tensor specification.

        Returns
        -------
        tf.TensorSpec
            Output tensor specification.
        """
        return inputs_spec

    def get_config(self):
        """Get the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_attention_heads": self.num_attention_heads,
                "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            }
        )
        return config
