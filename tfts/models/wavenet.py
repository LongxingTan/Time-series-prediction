"""
`WaveNet: A Generative Model for Raw Audio
<https://arxiv.org/abs/1609.03499>`_
"""

import logging
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Lambda, ReLU

from tfts.generation import GenerationEngine, MeanSampler, StepOutput
from tfts.generation.decoding import DecodeSession
from tfts.generation.feedback import FeedbackPolicy
from tfts.layers.cnn_layer import ConvTemp
from tfts.layers.dense_layer import DenseTemp

from ._autoregressive import AUTOREGRESSIVE_CAPABILITIES, AutoregressiveModel, decoder_features, encoder_features
from .base import CommonConfig
from .registry import register_model

logger = logging.getLogger(__name__)


class WaveNetConfig(CommonConfig):
    model_type: str = "wavenet"

    def __init__(
        self,
        dilation_rates: List[int] = None,
        kernel_sizes: List[int] = None,
        filters: int = 128,
        dense_hidden_size: int = 64,
        scheduled_sampling: float = 1.0,
        use_attention: bool = False,
        attention_size: int = 64,
        num_attention_heads: int = 2,
        attention_probs_dropout_prob: float = 0.0,
        target_dim: int = 1,
        **kwargs,
    ) -> None:
        """
        Initializes the configuration for the WaveNet model with the specified parameters.

        Args:
            dilation_rates: List of dilation rates for the convolutional layers.
            kernel_sizes: List of kernel sizes for the convolutional layers.
            filters: The number of filters in the convolutional layers.
            dense_hidden_size: The size of the dense hidden layer following the convolutional layers.
            scheduled_sampling: Scheduled sampling ratio. 0 means teacher forcing, 1 means use last prediction
            use_attention: Whether to use attention mechanism in the model.
            attention_size: The size of the attention mechanism.
            num_attention_heads: The number of attention heads.
            attention_probs_dropout_prob: Dropout probability for attention probabilities.
        """
        super(WaveNetConfig, self).__init__()

        self.target_dim = target_dim
        self.dilation_rates: List[int] = dilation_rates or [2**i for i in range(4)]
        self.kernel_sizes: List[int] = kernel_sizes or [2] * 4
        self.filters: int = filters
        self.dense_hidden_size: int = dense_hidden_size
        self.scheduled_sampling: float = scheduled_sampling
        self.use_attention: bool = use_attention
        self.attention_size: int = attention_size
        self.num_attention_heads: int = num_attention_heads
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob


@register_model(
    "wavenet",
    config=WaveNetConfig,
    paper="https://arxiv.org/abs/1609.03499",
    tags=("convolutional", "long-range"),
    capabilities=AUTOREGRESSIVE_CAPABILITIES,
)
class WaveNet(AutoregressiveModel):
    """WaveNet model for time series"""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[WaveNetConfig] = None) -> None:
        """
        Initializes the WaveNet model.

        Args:
            predict_sequence_length: Length of the prediction sequence.
            config: Configuration object containing model parameters.
        """
        super(WaveNet, self).__init__()
        self.config = config or WaveNetConfig()
        self.predict_sequence_length = predict_sequence_length
        self.encoder = Encoder(
            kernel_sizes=self.config.kernel_sizes,
            dilation_rates=self.config.dilation_rates,
            filters=self.config.filters,
            dense_hidden_size=self.config.dense_hidden_size,
        )
        self.decoder = Decoder(
            target_dim=self.config.target_dim,
            filters=self.config.filters,
            dilation_rates=self.config.dilation_rates,
            dense_hidden_size=self.config.dense_hidden_size,
            predict_sequence_length=self.predict_sequence_length,
        )

    def initialize_decode(self, batch, *, horizon, training=False):
        features = decoder_features(batch, horizon)
        _, memory = self.encoder(encoder_features(batch))
        if not self.decoder.built:
            self.decoder.build(features.shape)
        return DecodeSession(features, self.decoder.initialize_state(memory), self.decoder_seed(batch))

    def decode_step(self, previous, state, context, *, offset, training=False):
        return self.decoder.step(previous, state, context[:, offset, :])


class Encoder(tf.keras.layers.Layer):
    """Encoder block for the WaveNet model."""

    def __init__(
        self, kernel_sizes: List[int], filters: int, dilation_rates: List[int], dense_hidden_size: int, **kwargs
    ) -> None:
        """
        Initializes the encoder block.

        Args:
            kernel_sizes: List of kernel sizes for convolutional layers.
            filters: Number of filters for convolutional layers.
            dilation_rates: Dilation rates for the convolutions.
            dense_hidden_size: Hidden size for the dense layers.
        """
        super(Encoder, self).__init__(**kwargs)
        self.filters = filters
        self.conv_times = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilation_rates)):
            self.conv_times.append(
                ConvTemp(filters=2 * filters, kernel_size=kernel_size, causal=True, dilation_rate=dilation)
            )
        self.dense_time1 = DenseTemp(hidden_size=filters, activation="tanh", name="encoder_dense_time1")
        self.dense_time2 = DenseTemp(hidden_size=filters + filters, name="encoder_dense_time2")
        self.dense_time3 = DenseTemp(hidden_size=dense_hidden_size, activation="relu", name="encoder_dense_time3")
        self.dense_time4 = DenseTemp(hidden_size=1, name="encoder_dense_time_4")

    def call(self, x: tf.Tensor):
        inputs = self.dense_time1(inputs=x)

        skip_outputs = []
        conv_inputs = [inputs]
        for conv_time in self.conv_times:
            dilated_conv = conv_time(inputs)
            split_layer = Lambda(lambda x: tf.split(x, 2, axis=2))
            conv_filter, conv_gate = split_layer(dilated_conv)
            dilated_conv = Lambda(lambda x: tf.nn.tanh(x[0]) * tf.nn.sigmoid(x[1]))([conv_filter, conv_gate])
            outputs = self.dense_time2(inputs=dilated_conv)
            split_layer2 = Lambda(lambda x: tf.split(x, [self.filters, self.filters], axis=2))
            skips, residuals = split_layer2(outputs)
            inputs += residuals
            conv_inputs.append(inputs)  # batch_size * time_sequence_length * filters
            skip_outputs.append(skips)

        concat_layer = Concatenate(axis=2)
        concatenated = concat_layer(skip_outputs)
        relu_layer = ReLU()
        skip_outputs = relu_layer(concatenated)
        # skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = self.dense_time3(skip_outputs)
        # [batch_size, time_sequence_length, filters] * time_sequence_length
        y_hat = self.dense_time4(h)
        return y_hat, conv_inputs[:-1]


class Decoder(tf.keras.layers.Layer):
    """Kernel-two decoder with bounded, explicit per-layer delay buffers.

    Encoder kernels describe the history encoder. Decoder transitions retain
    their own weights and consume exactly one delayed activation per layer.
    """

    def __init__(self, filters, dilation_rates, dense_hidden_size, predict_sequence_length=24, target_dim=1, **kwargs):
        super().__init__(**kwargs)
        if not dilation_rates or any(d <= 0 for d in dilation_rates):
            raise ValueError("dilation_rates must contain positive integers")
        self.filters = filters
        self.dilation_rates = list(dilation_rates)
        self.dense_hidden_size = dense_hidden_size
        self.predict_sequence_length = predict_sequence_length
        self.target_dim = target_dim
        self.dense1 = Dense(filters, activation="tanh")
        self.dense2 = Dense(2 * filters, use_bias=True)
        self.dense3 = Dense(2 * filters, use_bias=False)
        self.dense4 = Dense(2 * filters)
        self.dense5 = Dense(dense_hidden_size, activation="relu")
        self.dense6 = Dense(target_dim)

    def build(self, input_shape):
        batch = input_shape[0]
        self.dense1.build([batch, input_shape[-1] + self.target_dim])
        for layer in (self.dense2, self.dense3, self.dense4):
            layer.build([batch, self.filters])
        self.dense5.build([batch, self.filters * len(self.dilation_rates)])
        self.dense6.build([batch, self.dense_hidden_size])
        super().build(input_shape)

    def initialize_state(self, memory):
        if len(memory) < len(self.dilation_rates):
            raise ValueError("one encoder buffer is required per decoder layer")
        buffers = []
        for values, dilation in zip(memory, self.dilation_rates):
            padding = tf.maximum(0, dilation - tf.shape(values)[1])
            buffers.append(tf.pad(values, [[0, 0], [padding, 0], [0, 0]])[:, -dilation:, :])
        return tuple(buffers)

    def step(self, previous, state, features):
        x = self.dense1(tf.concat([previous[:, 0, :], features], axis=-1))
        skips, buffers = [], []
        for buffer in state:
            filtered, gate = tf.split(self.dense2(buffer[:, 0, :]) + self.dense3(x), 2, axis=-1)
            skip, residual = tf.split(self.dense4(tf.tanh(filtered) * tf.sigmoid(gate)), 2, axis=-1)
            # Store this layer's input, not the next layer's residual output.
            buffers.append(tf.concat([buffer[:, 1:, :], x[:, None, :]], axis=1))
            x = x + residual
            skips.append(skip)
        value = self.dense6(self.dense5(tf.nn.relu(tf.concat(skips, axis=-1))))
        return StepOutput(value[:, None, :], state=tuple(buffers))

    def call(
        self, decoder_features, decoder_init_input, encoder_outputs, teacher=None, scheduled_sampling=0.0, training=None
    ):
        def step(previous, state, offset):
            return self.step(previous, state, decoder_features[:, offset, :])

        return (
            GenerationEngine(MeanSampler())
            .run(
                step,
                decoder_init_input[:, None, :],
                self.initialize_state(encoder_outputs),
                self.predict_sequence_length,
                teacher=teacher,
                feedback_policy=FeedbackPolicy(1.0 - scheduled_sampling if teacher is not None else 0.0),
            )
            .predictions
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                name: getattr(self, name)
                for name in (
                    "filters",
                    "dilation_rates",
                    "dense_hidden_size",
                    "predict_sequence_length",
                    "target_dim",
                )
            }
        )
        return config


DecoderV1 = Decoder
DecoderV2 = Decoder
