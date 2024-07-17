from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape: Tuple[Optional[int], ...]):
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x, masking=True):
        """
        Applies positional encoding to the input tensor.

        Parameters:
        x (tf.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        masking (bool, optional): If True, applies masking to the output tensor, by default True.

        Returns:
        tf.Tensor: Output tensor of the same shape as the input tensor, after applying positional encoding.
        """
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic

        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])  # => batch_size * seq_length
        position_enc = np.array(
            [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)] for pos in range(self.max_len)]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        if masking:
            outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 10000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len

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
        E = x.get_shape().as_list()[-1]  # static
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]  # dynamic
        with tf.name_scope("position_encode"):
            # # => batch_size * seq_length
            position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
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
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RotaryEmbedding(tf.keras.layers.Layer):
    """
    RoFormer: Enhanced Transformer with Rotary Position Embedding
    """

    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim

    def __call__(self, t, cache_key=None):
        return
