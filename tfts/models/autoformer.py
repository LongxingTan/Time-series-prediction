"""
`Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
<https://arxiv.org/abs/2106.13008>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization, ReLU

from tfts.layers.attention_layer import Attention, SelfAttention
from tfts.layers.autoformer_layer import AutoCorrelation, SeriesDecomp
from tfts.layers.dense_layer import FeedForwardNetwork
from tfts.layers.embed_layer import DataEmbedding

from ..layers.util_layer import ShapeLayer
from .base import BaseConfig, BaseModel


class AutoFormerConfig(BaseConfig):
    """
    Configuration class to store the configuration of a [`AutoFormer`]
    """

    model_type: str = "autoformer"

    def __init__(
        self,
        kernel_size=25,
        hidden_size=64,
        num_layers=1,
        num_decoder_layers=None,
        num_attention_heads=4,
        ffn_intermediate_size=256,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        positional_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.positional_type = positional_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class AutoFormer(BaseModel):
    """AutoFormer model"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[AutoFormerConfig] = None) -> None:
        super().__init__()
        self.config = config or AutoFormerConfig()
        self.predict_sequence_length = predict_sequence_length
        self.shape_layer = ShapeLayer()
        self.series_decomp = SeriesDecomp(self.config.kernel_size)
        self.encoder = Encoder(
            kernel_size=self.config.kernel_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )

        self.decoder = Decoder(
            kernel_size=self.config.kernel_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_decoder_layers,
            num_attention_heads=self.config.num_attention_heads,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            ffn_intermediate_size=self.config.ffn_intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
        )

        self.drop1 = Dropout(self.config.hidden_dropout_prob)
        self.dense1 = Dense(512, activation="relu")
        self.drop2 = Dropout(self.config.hidden_dropout_prob)
        self.dense2 = Dense(1024, activation="relu")
        self.project1 = Dense(1, activation=None)

    def __call__(
        self,
        inputs: tf.Tensor,
        teacher: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass of the AutoFormer model.

        Args:
            inputs: Input tensor(s) or dictionary.
                Can be:
                - A single tensor of shape [batch_size, time_steps, features]
                - A list/tuple of (x, encoder_feature, decoder_feature)
                - A dictionary with keys 'x' and 'encoder_feature'
            teacher: Optional teacher forcing input.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dictionary or tensor as output.

        Returns:
            If return_dict is True, returns a dictionary with model outputs.
            Otherwise, returns the output tensor.
        """
        x, encoder_feature, decoder_feature = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=False)
        # batch_size, _, n_feature = self.shape_layer(encoder_feature)

        # Encoder
        encoder_output = self.encoder(x)
        encoder_output = self.dense1(encoder_output)
        encoder_output = self.dense2(encoder_output)

        # Decoder
        decoder_output = self.decoder(decoder_feature, encoder_output)
        outputs = self.project1(decoder_output)
        return outputs


class Encoder(tf.keras.layers.Layer):
    """Encoder for Autoformer architecture."""

    def __init__(
        self,
        kernel_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def build(self, input_shape):
        super().build(input_shape)
        self.layers = [
            EncoderLayer(
                kernel_size=self.kernel_size,
                d_model=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                dropout_rate=self.hidden_dropout_prob,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = LayerNormalization()
        self.norm.build(list(input_shape[:-1]) + [self.hidden_size])
        self.built = True

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Process input through the encoder.

        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
            mask: Optional attention mask

        Returns:
            Processed tensor after applying encoder operations
        """
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size, time_steps, _ = input_shape
        return (batch_size, time_steps, self.hidden_size)


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder Layer for Autoformer architecture."""

    def __init__(
        self, kernel_size: int, d_model: int, num_attention_heads: int, dropout_rate: float = 0.1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        super().build(input_shape)
        self.series_decomp1 = SeriesDecomp(self.kernel_size)
        self.series_decomp2 = SeriesDecomp(self.kernel_size)
        self.autocorrelation = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.drop = Dropout(self.dropout_rate)
        self.dense = Dense(input_shape[-1])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.built = True

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Process input through the encoder layer.

        Args:
            x: Input tensor of shape [batch_size, time_steps, features]

        Returns:
            Processed tensor after applying encoder operations
        """
        # First sub-layer
        residual = x
        x = self.autocorrelation(x, x, x)
        x = self.drop(x)
        x = x + residual
        x = self.norm1(x)

        # Second sub-layer
        residual = x
        x = self.dense(x)
        x = self.drop(x)
        x = x + residual
        x = self.norm2(x)

        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder for Autoformer architecture."""

    def __init__(
        self,
        kernel_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ffn_intermediate_size: int,
        hidden_dropout_prob: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob

    def build(self, input_shape):
        super().build(input_shape)
        self.layers = [
            DecoderLayer(
                kernel_size=self.kernel_size,
                d_model=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                drop_rate=self.hidden_dropout_prob,
            )
            for _ in range(self.num_layers)
        ]
        self.norm = LayerNormalization()
        self.norm.build(list(input_shape[:-1]) + [self.hidden_size])
        self.built = True

    def call(
        self,
        x: tf.Tensor,
        memory: tf.Tensor,
        x_mask: Optional[tf.Tensor] = None,
        memory_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Process input through the decoder.

        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
            memory: Memory tensor from encoder
            x_mask: Optional attention mask for decoder input
            memory_mask: Optional attention mask for encoder memory

        Returns:
            Processed tensor after applying decoder operations
        """
        for layer in self.layers:
            x = layer(x, memory)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size, time_steps, _ = input_shape
        return (batch_size, time_steps, self.hidden_size)


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder Layer for Autoformer architecture."""

    def __init__(
        self, kernel_size: int, d_model: int, num_attention_heads: int, drop_rate: float = 0.1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.drop_rate = drop_rate

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.series_decomp1 = SeriesDecomp(self.kernel_size)
        self.series_decomp2 = SeriesDecomp(self.kernel_size)
        self.series_decomp3 = SeriesDecomp(self.kernel_size)
        self.autocorrelation1 = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.autocorrelation2 = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.conv1 = Conv1D(self.d_model, kernel_size=3, strides=1, padding="same")
        self.project = Conv1D(1, kernel_size=3, strides=1, padding="same")
        self.drop = Dropout(self.drop_rate)
        self.dense1 = Dense(input_shape[-1])
        self.conv2 = Conv1D(input_shape[-1], kernel_size=3, strides=1, padding="same")
        self.activation = ReLU()
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()
        super().build(input_shape)

    def call(self, x: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """Process input through the decoder layer.

        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
            memory: Memory tensor from encoder

        Returns:
            Processed tensor after applying decoder operations
        """
        # Self-attention sub-layer
        residual = x
        x = self.autocorrelation1(x, x, x)
        x = self.drop(x)
        x = x + residual
        x = self.norm1(x)

        # Cross-attention sub-layer
        residual = x
        x = self.autocorrelation2(x, memory, memory)
        x = self.drop(x)
        x = x + residual
        x = self.norm2(x)

        # Feed-forward sub-layer
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x + residual
        x = self.norm3(x)
        return x
