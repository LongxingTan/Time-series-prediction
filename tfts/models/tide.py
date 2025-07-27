"""
`Long-term Forecasting with TiDE: Time-series Dense Encoder
<https://arxiv.org/pdf/2304.08424v1.pdf>`_
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from ..layers.attention_layer import Attention, SelfAttention
from ..layers.dense_layer import FeedForwardNetwork
from ..layers.embed_layer import DataEmbedding
from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


class TideConfig(BaseConfig):
    model_type: str = "tide"

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        ffn_intermediate_size: int = 256,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.update(kwargs)


class Tide(BaseModel):
    """TiDE model for time series forecasting"""

    def __init__(self, predict_sequence_length=1, config: Optional[TideConfig] = None):
        super(Tide, self).__init__()
        self.config = config or TideConfig()
        self.predict_sequence_length = predict_sequence_length

        # Embedding layer
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")

        # Dense encoder layers
        self.dense_layers = []
        for _ in range(self.config.num_layers):
            self.dense_layers.append(
                DenseEncoderBlock(
                    hidden_size=self.config.hidden_size,
                    ffn_intermediate_size=self.config.ffn_intermediate_size,
                    dropout_rate=self.config.hidden_dropout_prob,
                    layer_norm_eps=self.config.layer_norm_eps,
                )
            )

        # Output projection
        self.output_projection = Dense(1)

    def __call__(self, x: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
        """Process inputs through the TiDE model.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, sequence_length, features).
        output_hidden_states : bool, optional
            Whether to output hidden states, by default None.
        return_dict : bool, optional
            Whether to return a dictionary of outputs, by default None.

        Returns
        -------
        tf.Tensor
            Output tensor of shape (batch_size, predict_sequence_length, 1).
        """
        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Embed inputs
        embedded = self.embedding(encoder_feature)

        # Process through dense encoder layers
        for layer in self.dense_layers:
            embedded = layer(embedded)

        # Project to output
        output = self.output_projection(embedded)

        # Slice the output to only include the last predict_sequence_length steps
        output = output[:, -self.predict_sequence_length :, :]

        return output


class DenseEncoderBlock(tf.keras.layers.Layer):
    """Dense encoder block with feed-forward networks and layer normalization."""

    def __init__(
        self,
        hidden_size: int,
        ffn_intermediate_size: int,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-9,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_intermediate_size, dropout_rate)
        self.layernorm = LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        ffn_output = self.ffn(inputs)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.layernorm(inputs + ffn_output)
