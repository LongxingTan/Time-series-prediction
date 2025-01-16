"""
`RWKV: Reinventing RNNs for the Transformer Era
<https://arxiv.org/abs/2305.13048>`_
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import GRU, BatchNormalization, Dense, Dropout

from tfts.layers.attention_layer import Attention

from .base import BaseConfig, BaseModel


class RWKVConfig(BaseConfig):
    model_type: str = "rwkv"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        num_stacked_layers: int = 1,
        attention_heads: int = 8,
        attention_size: int = 64,
        dense_hidden_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the configuration for the RWKV model with the specified parameters.

        Args:
            num_stacked_layers: The number of stacked RWKV layers.
            attention_heads: Number of attention heads for the mechanism.
            attention_size: Size of each attention head.
            dense_hidden_size: The size of the dense hidden layer following the RWKV.
        """
        super().__init__()

        self.block_size: int = block_size
        self.vocab_size: int = vocab_size
        self.num_stacked_layers: int = num_stacked_layers
        self.attention_heads: int = attention_heads
        self.attention_size: int = attention_size
        self.dense_hidden_size: int = dense_hidden_size
        self.dropout: float = dropout


class RWKV(BaseModel):
    """TensorFlow RWKV model"""

    def __init__(self, predict_sequence_length: int = 1, config=None):
        super().__init__(config)
        if config is None:
            config = RWKVConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length
        self.encoder = RWKVEncoder(
            rnn_size=config.rnn_hidden_size,
            attention_heads=config.attention_heads,
            attention_size=config.attention_size,
            dense_size=config.dense_hidden_size,
        )
        self.project1 = Dense(predict_sequence_length, activation=None)

        self.dense1 = Dense(128, activation="relu")
        self.bn = BatchNormalization()
        self.dense2 = Dense(128, activation="relu")

    def __call__(self, inputs, teacher=None, return_dict: Optional[bool] = None):
        """RWKV model call"""

        x, encoder_feature = self._prepare_inputs(inputs)
        encoder_outputs, encoder_state = self.encoder(encoder_feature)

        encoder_output = self.dense1(encoder_state)
        encoder_output = self.dense2(encoder_output)

        outputs = self.project1(encoder_output)

        expand_dims_layer = tf.keras.layers.Reshape((outputs.shape[1], 1))
        outputs = expand_dims_layer(outputs)

        return outputs

    def _prepare_inputs(self, inputs):
        """Prepare the inputs for the encoder."""

        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, _ = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            x = inputs
            encoder_feature = x
        return x, encoder_feature


class RWKVEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn_size: int,
        attention_heads: int,
        attention_size: int,
        dense_size: int,
        num_stacked_layers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        self.dense_size = dense_size
        self.num_stacked_layers = num_stacked_layers

        self.attention_layers = [
            Attention(hidden_size=self.attention_size, num_attention_heads=self.attention_heads)
            for _ in range(self.num_stacked_layers)
        ]

    def call(self, inputs: tf.Tensor):
        """RWKV Encoder call"""

        x = inputs
        for layer in self.attention_layers:
            x = layer(x)

        return x, x  # Placeholder for encoder state (could be modified for your use case)

    def get_config(self):
        config = {
            "rnn_size": self.rnn_size,
            "attention_heads": self.attention_heads,
            "attention_size": self.attention_size,
            "dense_size": self.dense_size,
        }
        base_config = super(RWKVEncoder, self).get_config()
        return {**base_config, **config}
