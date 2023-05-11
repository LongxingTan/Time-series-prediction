# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.transformer`"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Conv1D, Dense, Dropout, Embedding, LayerNormalization, SpatialDropout1D


class TokenEmbedding(tf.keras.layers.Layer):
    """
    A layer that performs token embedding.

    Args:
        embed_size (int): The size of the embedding.

    Input shape:
        - 3D tensor with shape `(batch_size, time_steps, input_dim)`

    Output shape:
        - 3D tensor with shape `(batch_size, time_steps, embed_size)`
    """

    def __init__(self, embed_size: int):
        super(TokenEmbedding, self).__init__()
        self.embed_size = embed_size

    def build(self, input_shape):
        self.token_weights = self.add_weight(
            name="token_weights",
            shape=[input_shape[-1], self.embed_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=self.embed_size**-0.5),
        )
        super(TokenEmbedding, self).build(input_shape)

    def call(self, x):
        """
        Performs the token embedding.

        Args:
            x (tensor): Input tensor.

        Returns:
            Tensor: Embedded tensor.
        """
        y = tf.einsum("bsf,fk->bsk", x, self.token_weights)
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(TokenEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TokenRnnEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_size: int) -> None:
        super().__init__()
        self.embed_size = embed_size

    def build(self, input_shape):
        self.rnn = GRU(self.embed_size, return_sequences=True, return_state=True)
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape `(batch_size, sequence_length, input_dim)`.

        Returns
        -------
        y : tf.Tensor
            Output tensor of shape `(batch_size, sequence_length, embed_size)`.
        """
        y, _ = self.rnn(x)
        return y

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(TokenRnnEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape):
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
    def __init__(self, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len

    def build(self, input_shape):
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
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            if masking:
                outputs = tf.where(tf.equal(x, 0), x, outputs)
        return tf.cast(outputs, tf.float32)

    def get_config(self):
        config = {"max_len": self.max_len}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def build(self, input_shape):
        self.embed = tf.keras.layers.Embedding(input_dim=input_shape[1], output_dim=input_shape[2])
        super().build(input_shape)

    def call(self, x, **kwargs):
        """Perform the embedding lookup operation.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor of shape (batch_size, seq_length).
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tf.Tensor
            The tensor of embeddings of shape (batch_size, seq_length, embedding_size).
        """
        return self.embed(x)


class TemporalEmbedding(tf.keras.layers.Layer):
    # minute, hour, weekday, day, month
    def __init__(self, embed_size, embed_type="fixed") -> None:
        super().__init__()
        minute_size = 6  # every 10 minutes
        hour_size = 24  #
        self.minute_embed = Embedding(minute_size, 3)
        self.hour_embed = Embedding(hour_size, 6)

    def call(self, x, **kwargs):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        return


class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_size: int, dropout: float = 0.0):
        """
        Data Embedding layer.

        Args:
            embed_size (int): Embedding size for tokens.
            dropout (float, optional): Dropout rate to apply. Defaults to 0.0.
        """
        super(DataEmbedding, self).__init__()
        self.embed_size = embed_size
        self.value_embedding = TokenEmbedding(embed_size)
        self.positional_embedding = PositionalEncoding()
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(DataEmbedding, self).build(input_shape)

    def call(self, x):
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_length).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, seq_length, embed_size).
        """
        ve = self.value_embedding(x)
        pe = self.positional_embedding(ve)
        return self.dropout(ve + pe)

    def get_config(self):
        config = {"embed_size": self.embed_size}
        base_config = super(DataEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
