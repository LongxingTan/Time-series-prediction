"""
`A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
<https://arxiv.org/abs/2211.14730>`_
"""

from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization

from tfts.layers.attention_layer import Attention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

from ..layers.util_layer import ShapeLayer
from .base import BaseConfig, BaseModel


class PatchTSTConfig(BaseConfig):
    model_type: str = "patch_tst"

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
        patch_size: int = 16,
        **kwargs
    ) -> None:
        """
        Initializes the configuration for the PatchTST model with the specified parameters.

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
            patch_size: Size of each patch for time series segmentation.
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
        self.patch_size: int = patch_size
        self.update(kwargs)


class PatchTST(BaseModel):
    """TensorFlow PatchTST model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[PatchTSTConfig] = None):
        super().__init__()
        self.config = config or PatchTSTConfig()
        self.predict_sequence_length = predict_sequence_length

        # Embedding layer
        self.embedding = DataEmbedding(self.config.hidden_size, positional_type="positional encoding")

        # Patch embedding
        self.patch_embedding = Dense(self.config.hidden_size)

        # Transformer blocks
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]

        # Output projection
        self.output_projection = Dense(1)

    def __call__(
        self,
        x,
        states=None,
        teacher=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """PatchTST model call for time series forecasting"""

        # Prepare inputs
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(x, ignore_decoder_inputs=False)

        # Create patches
        batch_size, seq_length, _ = ShapeLayer()(encoder_feature)
        num_patches = seq_length // self.config.patch_size

        # Reshape to patches
        patches = tf.reshape(
            encoder_feature[:, : num_patches * self.config.patch_size, :],
            [batch_size, num_patches, self.config.patch_size, -1],
        )

        # Flatten patches and project to hidden size
        patches = tf.reshape(patches, [batch_size, num_patches, -1])
        x = self.patch_embedding(patches)

        # Add positional embeddings
        x = self.embedding(x)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Project to output
        x = self.output_projection(x)

        # Reshape back to original sequence length
        x = tf.reshape(x, [batch_size, -1, 1])

        # Slice the output to only include the last predict_sequence_length steps
        x = x[:, -self.predict_sequence_length :, :]

        return x


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block for PatchTST"""

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

    def call(self, x):
        """Transformer block forward pass"""
        # Self-attention
        attention_output = self.attention(x, x, x)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        x = self.attention_norm(x + attention_output)

        # Feed-forward
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        x = self.feed_forward_norm(x + feed_forward_output)

        return x
