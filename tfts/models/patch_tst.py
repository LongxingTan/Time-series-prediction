"""
`A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
<https://arxiv.org/abs/2211.14730>`_
"""

from typing import Dict, Optional

try:
    from keras import ops
except ImportError:  # Keras 2 bundled with TensorFlow < 2.16
    ops = None

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LayerNormalization

from tfts.layers.attention_layer import Attention
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

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
        output_size: int = 1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        patch_size: int = 16,
        patch_stride: int = 8,
        residual_last_value: bool = False,
        **kwargs,
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
        self.output_size: int = output_size
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.layer_norm_eps: float = layer_norm_eps
        self.pad_token_id: int = pad_token_id
        self.patch_size: int = patch_size
        self.patch_stride: int = patch_stride
        self.residual_last_value: bool = residual_last_value
        self.update(kwargs)


class PatchTST(BaseModel):
    """TensorFlow PatchTST model for time series forecasting"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[PatchTSTConfig] = None):
        super().__init__()
        self.config = config or PatchTSTConfig()
        self.predict_sequence_length = predict_sequence_length

        # PatchTST embeds each variable independently; the batch and channel
        # axes are folded together before the shared encoder.
        self.patch_embedding = Dense(self.config.hidden_size)
        self.position_embedding = tf.keras.layers.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )

        # Transformer blocks
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]

        # Output projection
        self.flatten = Flatten()
        self.output_projection = Dense(self.predict_sequence_length)

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

        # Reversible instance normalization, matching the reference forecast
        # path. Statistics are detached from the gradient.
        means = tf.stop_gradient(tf.reduce_mean(encoder_feature, axis=1, keepdims=True))
        normalized = encoder_feature - means
        stdev = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(normalized), axis=1, keepdims=True) + 1e-5))
        normalized = normalized / stdev

        seq_length = int(encoder_feature.shape[1])
        channels = int(encoder_feature.shape[-1])
        target_channels = min(channels, self.config.output_size)
        padding = self.config.patch_stride
        # Reference PatchEmbedding pads by repeating the final observation.
        normalized = tf.pad(normalized, [[0, 0], [0, padding], [0, 0]], mode="SYMMETRIC")
        channel_series = tf.transpose(normalized, [0, 2, 1])
        patches = tf.signal.frame(
            channel_series,
            frame_length=self.config.patch_size,
            frame_step=self.config.patch_stride,
            axis=-1,
        )
        num_patches = int((seq_length - self.config.patch_size) / self.config.patch_stride + 2)
        reshape = ops.reshape if ops is not None else tf.reshape
        patches = reshape(patches, (-1, num_patches, self.config.patch_size))
        x = self.patch_embedding(patches)
        x = x + self.position_embedding(tf.range(num_patches))[None, :, :]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.output_projection(x)
        x = reshape(x, (-1, channels, self.predict_sequence_length))[:, :target_channels, :]
        x = tf.transpose(x, [0, 2, 1])
        location = (
            encoder_feature[:, -1:, :target_channels]
            if self.config.residual_last_value
            else means[:, :1, :target_channels]
        )
        return x * stdev[:, :1, :target_channels] + location


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
