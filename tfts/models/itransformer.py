"""
`iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
<https://arxiv.org/abs/2310.06625>`_
"""

from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization

from tfts.layers.attention_layer import Attention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

from ..layers.util_layer import ShapeLayer
from .base import BaseConfig, BaseModel


class ITransformerConfig(BaseConfig):
    model_type: str = "itransformer"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        ffn_intermediate_size: int = 256,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        **kwargs
    ) -> None:
        """
        Initializes the configuration for the iTransformer model with the specified parameters.

        Args:
            hidden_size: Size of each attention head.
            num_layers: The number of stacked transformer layers.
            num_attention_heads: The number of attention heads.
            attention_probs_dropout_prob: Dropout rate for attention probabilities.
            hidden_dropout_prob: Dropout rate for hidden layers.
            ffn_intermediate_size: Size of the intermediate layer in the feed-forward network.
            max_position_embeddings: Maximum sequence length for positional embeddings.
            initializer_range: Standard deviation for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: ID for padding token.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.num_attention_heads: int = num_attention_heads
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.ffn_intermediate_size: int = ffn_intermediate_size
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.pad_token_id: int = pad_token_id
        self.update(kwargs)


class ITransformer(BaseModel):
    """TensorFlow iTransformer model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[ITransformerConfig] = None):
        super().__init__()
        self.config = config or ITransformerConfig()
        self.predict_sequence_length = predict_sequence_length

        # In iTransformer, we embed the entire time dimension of each variate
        self.enc_embedding = Dense(self.config.hidden_size)

        # Transformer blocks
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]

        # Project from hidden_size to predict_sequence_length
        self.projector = Dense(self.predict_sequence_length)

    def __call__(self, x, training=None, **kwargs):
        """iTransformer forward pass: Inverting Variates and Time"""
        # x shape: (batch, seq_len, n_vars)
        x, encoder_feature, _ = self._prepare_3d_inputs(x, ignore_decoder_inputs=True)

        # 1. Inversion: (batch, seq_len, n_vars) -> (batch, n_vars, seq_len)
        # Each variate becomes a "token"
        x = tf.transpose(encoder_feature, perm=[0, 2, 1])

        # 2. Embedding: Map the whole history of each variate to hidden_size
        # (batch, n_vars, hidden_size)
        x = self.enc_embedding(x)

        # 3. Attention: Process across variables
        for block in self.blocks:
            x = block(x, training=training)

        # 4. Projection: Map hidden_size to predict_len
        # (batch, n_vars, predict_len)
        x = self.projector(x)

        # 5. Reverse Inversion: (batch, predict_len, n_vars)
        x = tf.transpose(x, perm=[0, 2, 1])

        return x


class TransformerBlock(tf.keras.layers.Layer):
    """Standard Transformer block used in iTransformer"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        )
        self.attention_output = Dense(config.hidden_size)
        self.attention_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.attention_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.feed_forward = FeedForwardNetwork(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.feed_forward_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.feed_forward_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, x, training=None):
        # Self-attention across variates
        attention_output = self.attention(x, x, x, training=training)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output, training=training)
        x = self.attention_norm(x + attention_output)

        # Feed-forward
        ffn_output = self.feed_forward(x, training=training)
        ffn_output = self.feed_forward_dropout(ffn_output, training=training)
        x = self.feed_forward_norm(x + ffn_output)
        return x
