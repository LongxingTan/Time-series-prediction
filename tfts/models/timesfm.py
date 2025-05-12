"""
`A decoder-only foundation model for time-series forecasting
<https://arxiv.org/abs/2310.10688>`_
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


class TimesFmConfig(BaseConfig):
    model_type: str = "timesfm"

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


class TimesFm(BaseModel):
    """TimesFm model for time series forecasting"""

    def __init__(self, predict_sequence_length=1, config: Optional[TimesFmConfig] = None):
        super(TimesFm, self).__init__()
        self.config = config or TimesFmConfig()
        self.predict_sequence_length = predict_sequence_length

        # Embedding layer
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")

        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(self.config.num_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    ffn_intermediate_size=self.config.ffn_intermediate_size,
                    rate=self.config.hidden_dropout_prob,
                    layer_norm_eps=self.config.layer_norm_eps,
                )
            )

        # Output projection
        self.output_projection = Dense(1)

    def __call__(self, x: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
        """Process inputs through the TimesFm model.

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

        # Process through transformer blocks
        for block in self.transformer_blocks:
            embedded = block(embedded)

        # Project to output
        output = self.output_projection(embedded)

        # Slice the output to only include the last predict_sequence_length steps
        output = output[:, -self.predict_sequence_length :, :]

        return output


class TransformerBlock(tf.keras.layers.Layer):
    """Basic Transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_intermediate_size: int,
        rate: float = 0.1,
        layer_norm_eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, rate)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_intermediate_size, rate)
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
