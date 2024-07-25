"""Layer for :py:class:`~tfts.models.transformer`"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Conv1D, Dense, Dropout, Embedding, LayerNormalization, SpatialDropout1D

from .position_layer import PositionalEmbedding, PositionalEncoding


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

    def build(self, input_shape: Tuple[Optional[int], ...]):
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

    def build(self, input_shape: Tuple[Optional[int], ...]):
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

    def build(self, input_shape: Tuple[Optional[int], ...]):
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
