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


class AutoFormerConfig(object):
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


class AutoFormer(object):
    """AutoFormer model"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[AutoFormerConfig] = None) -> None:
        self.config = config or AutoFormerConfig()
        self.predict_sequence_length = predict_sequence_length

        # self.encoder_embedding = TokenEmbedding(config['hidden_size'])
        self.series_decomp = SeriesDecomp(config.kernel_size)
        self.encoder = [
            EncoderLayer(
                config.kernel_size,
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
            for _ in range(config.num_layers)
        ]

        self.decoder = [
            DecoderLayer(
                config.kernel_size,
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
            for _ in range(config.num_decoder_layers)
        ]

        self.project = Conv1D(1, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.project1 = Dense(predict_sequence_length, activation=None)
        self.drop1 = Dropout(0.0)
        self.dense1 = Dense(512, activation="relu")
        self.drop2 = Dropout(0.0)
        self.dense2 = Dense(1024, activation="relu")

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
            training: Whether the model is in training mode.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dictionary or tensor as output.
            
        Returns:
            If return_dict is True, returns a dictionary with model outputs.
            Otherwise, returns the output tensor.
        """
        if isinstance(inputs, (list, tuple)):
            x, encoder_feature, decoder_feature = inputs
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        elif isinstance(inputs, dict):
            x = inputs["x"]
            encoder_feature = inputs["encoder_feature"]
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:
            encoder_feature = x = inputs

        batch_size, _, n_feature = tf.shape(encoder_feature)
        # # de-comp
        # seasonal_init, trend_init = self.series_decomp(encoder_feature)
        # # decoder input
        # seasonal_init = tf.concat(
        #     [seasonal_init, tf.zeros([batch_size, self.predict_sequence_length, n_feature])], axis=1
        # )
        # trend_init = tf.concat(
        #     [
        #         trend_init,
        #         tf.repeat(
        #             tf.reduce_mean(encoder_feature, axis=1)[:, tf.newaxis, :],
        #             repeats=self.predict_sequence_length,
        #             axis=1,
        #         ),
        #     ],
        #     axis=1,
        # )

        # encoder
        for layer in self.encoder:
            x = layer(x)

        encoder_output = x[:, -1]
        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        # encoder_output = self.bn2(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)
        outputs = tf.expand_dims(outputs, -1)
        return outputs

        # encoder_output = x
        # trend_part= trend_init[..., 0:1]

        # for layer in self.decoder:
        #     seasonal_part, trend_part = layer(seasonal_init, encoder_output, trend_part)

        # trend_part = trend_part[:, -self.predict_sequence_length:, :]
        # seasonal_part = seasonal_part[:, -self.predict_sequence_length:, :]
        # seasonal_part = self.project(seasonal_part)
        # return trend_part + seasonal_part


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder Layer for Autoformer architecture.
    
    This layer implements the encoder block of the Autoformer architecture,
    which combines auto-correlation attention with series decomposition.
    
    Attributes:
        kernel_size: Size of the kernel for series decomposition.
        d_model: The dimension of the model/embedding.
        num_attention_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
    """
    def __init__(self, kernel_size: int, d_model: int, num_attention_heads: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.series_decomp1 = SeriesDecomp(self.kernel_size)
        self.series_decomp2 = SeriesDecomp(self.kernel_size)
        self.autocorrelation = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.drop = Dropout(self.dropout_rate)
        self.dense = Dense(input_shape[-1])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Process input through the encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
            training: Whether the layer is being called during training
            
        Returns:
            Processed tensor after applying encoder operations
        """
        x, _ = self.series_decomp1(self.autocorrelation(x, x, x) + x)
        x, _ = self.series_decomp2(self.drop(self.dense(x)) + x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder Layer for Autoformer architecture.
    
    This layer implements the decoder block of the Autoformer architecture,
    which combines auto-correlation attention with series decomposition.
    
    Attributes:
        kernel_size: Size of the kernel for series decomposition.
        d_model: The dimension of the model/embedding.
        num_attention_heads: Number of attention heads.
        drop_rate: Dropout rate for regularization.
    """
    def __init__(self, kernel_size, d_model, num_attention_heads, drop_rate=0.1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.drop_rate = drop_rate

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer with input shape information."""
        self.series_decomp1 = SeriesDecomp(self.kernel_size, name="series_decomp1")
        self.series_decomp2 = SeriesDecomp(self.kernel_size, name="series_decomp2")
        self.series_decomp3 = SeriesDecomp(self.kernel_size, name="series_decomp3")
        self.autocorrelation1 = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.autocorrelation2 = AutoCorrelation(self.d_model, self.num_attention_heads)
        self.conv1 = Conv1D(self.d_model, kernel_size=3, strides=1, padding="same")
        self.project = Conv1D(1, kernel_size=3, strides=1, padding="same")
        self.drop = Dropout(self.drop_rate)
        self.dense1 = Dense(input_shape[-1])
        self.conv2 = Conv1D(input_shape[-1], kernel_size=3, strides=1, padding="same")
        self.activation = ReLU()

    def call(self, x, cross, init_trend):
        """Process inputs through the decoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, time_steps, features]
            cross: Cross attention tensor of shape [batch_size, time_steps, features]
            init_trend: Initial trend component of shape [batch_size, time_steps, 1]
            training: Whether the layer is being called during training
            
        Returns:
            Tuple of (seasonal component, updated trend component)
        """
        x, trend1 = self.series_decomp1(self.drop(self.autocorrelation1(x, x, x)) + x)
        x, trend2 = self.series_decomp2(self.drop(self.autocorrelation2(x, cross, cross)) + x)
        x = self.conv2(self.drop(self.activation(self.conv1(x))))
        x, trend3 = self.series_decomp3(self.drop(self.dense1(x)) + x)

        trend = trend1 + trend2 + trend3
        trend = self.drop(self.project(trend))
        final_trend = init_trend + trend
        return x, final_trend
