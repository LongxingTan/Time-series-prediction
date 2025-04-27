"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x, masking: bool = True):
        """Applies positional encoding to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            A tensor of shape (batch_size, train_sequence_length, input_size)
        masking :  bool, optional
            If True, applies masking to the output tensor, by default True.

        Returns
        -------
        outputs: tf.Tensor
            Output tensor of the same shape as the input tensor, after applying positional encoding.
        """
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # => batch_size * seq_length
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (max_len, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len  # TODO: check if without this works

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x, masking=True):
        """Applies positional encoding to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        masking : bool, optional
            Whether to mask padded values, by default True.

        Returns
        -------
        tf.Tensor
            The output tensor of shape (batch_size, seq_length, embed_dim) with positional encoding applied.
        """

        d_model = x.get_shape().as_list()[-1]  # static
        depth = d_model // 2
        _, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        with tf.name_scope("positional_encode"):
            # => (max_len, 1)
            positions = tf.range(seq_length, dtype=tf.float32)[..., tf.newaxis]
            # => (1, d_model/2)
            depths = tf.range(depth, dtype=tf.float32)[np.newaxis, :] / depth
            # => (1, d_model/2)
            angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
            # => (max_len, d_model/2)
            angle_rads = tf.linalg.matmul(positions, angle_rates)
            # => (max_len, d_model)
            position_enc = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)

        return position_enc

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len=512, output_dim=512):
        super(RelativePositionEmbedding, self).__init__()
        self.max_len = max_len
        self.output_dim = output_dim

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embedding_initializer = tf.keras.initializers.get("zeros")
        self.embeddings = self.add_weight(
            name="RelativePositionEmbedding",
            shape=(self.max_len, self.output_dim),
            initializer=self.embedding_initializer,
        )

    def call(self, inputs):
        """relative position embedding

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        """
        q, v = inputs
        q_idx = tf.range(0, tf.shape(q)[1], dtype=tf.int32)
        q_idx = tf.expand_dims(q_idx, 1)
        v_idx = tf.range(0, tf.shape(v)[1], dtype=tf.int32)
        v_idx = tf.expand_dims(v_idx, 0)

        position_idx = v_idx - q_idx
        max_position = (self.input_dim - 1) // 2
        position_idx = tf.clip_by_value(position_idx, -max_position, max_position)
        position_idx = position_idx + max_position
        return tf.gather(self.embeddings, position_idx)


class RotaryPositionEmbedding(tf.keras.layers.Layer):
    """
    RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/src/layers/modeling/rotary_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, inputs, cache_key=None):
        """rotary position embedding

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor of shape (batch_size, seq_length, embed_dim).
        """
        return inputs
