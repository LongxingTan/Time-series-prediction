"""Layer for :py:class:`~tfts.models.autoformer`"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

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
        self.avg = AveragePooling1D(pool_size=kernel_size, strides=stride, padding="valid")

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
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = MovingAvg(kernel_size, stride=1)

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
    """Self-Attention layer that computes time-delayed autocorrelation between queries and keys"""

    def __init__(self, d_model: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.0) -> None:
        super().__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError(f"Hidden size {d_model} must be divisible by the number of heads {num_attention_heads}.")
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_size = d_model // num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.wq = Dense(self.d_model, name="q")
        self.wk = Dense(self.d_model, name="k")
        self.wv = Dense(self.d_model, name="v")
        self.drop = Dropout(self.attention_probs_dropout_prob)
        self.dense = Dense(self.d_model, name="project")
        super().build(input_shape)

    def time_delay_agg(self, q, k, v):  # TODO: v not used in process
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
        Tensor of shape (batch_size, num_attention_heads, time_steps, hidden_size)
            Time-delayed autocorrelation between queries and keys.
        """
        batch_size = tf.shape(q)[0]
        time_steps = tf.shape(q)[2]
        q_fft = tf.signal.rfft(tf.transpose(q, perm=[0, 1, 3, 2]))
        k_fft = tf.signal.rfft(tf.transpose(k, perm=[0, 1, 3, 2]))
        S_qk = q_fft * tf.math.conj(k_fft)
        R_qk = tf.signal.irfft(S_qk)

        init_index = tf.reshape(tf.range(time_steps), (1, 1, 1, -1))
        init_index = tf.tile(init_index, [batch_size, self.num_attention_heads, self.hidden_size, 1])
        top_k = int(2 * tf.math.log(tf.cast(time_steps, tf.float32)))
        # mean_value = tf.reduce_mean(R_qk, axis=1)
        weights, indices = tf.math.top_k(R_qk, top_k)

        tmp_corr = tf.nn.softmax(weights, axis=-1)
        # tmp_corr = tf.reshape(tmp_corr, (batch_size, 1, 32, -1))

        tmp_values = tf.tile(tf.transpose(q, perm=[0, 1, 3, 2]), tf.constant([1, 1, 1, 2]))
        delays_agg = tf.zeros_like(tf.transpose(q, perm=[0, 1, 3, 2]))

        for i in range(top_k):
            pattern = tf.gather(tmp_values, init_index + tf.expand_dims(indices[..., i], -1), axis=-1, batch_dims=-1)
            delays_agg = delays_agg + pattern * (tf.expand_dims(tmp_corr[..., i], axis=-1))
        return delays_agg

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_attention_heads, timesteps, depth)

    def call(self, q, k, v, dynamic=True):
        """_summary_

        Parameters
        ----------
        q : _type_
            _description_
        k : _type_
            _description_
        v : _type_
            _description_
        dynamic : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        batch_size = tf.shape(q)[0]

        q = self.drop(self.wq(q))
        k = self.drop(self.wk(k))
        v = self.drop(self.wv(v))

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        L = tf.shape(q)[2]
        S = tf.shape(v)[2]

        if tf.math.greater(L, S):
            zeros = tf.zeros_like(q[:, :, : (L - S), :])
            v = tf.concat([v, zeros], axis=2)
            k = tf.concat([k, zeros], axis=2)
        else:
            v = v[:, :, :L, :]
            k = k[:, :, :L, :]

        delays_agg = self.time_delay_agg(q, k, v)
        delays_agg = tf.transpose(delays_agg, [0, 3, 1, 2])
        concat_delays_agg = tf.reshape(delays_agg, (batch_size, -1, self.d_model))
        output = self.dense(concat_delays_agg)
        return output
