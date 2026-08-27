"""
`TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
<https://arxiv.org/abs/2210.02186>`_

Key idea: instead of modeling the full 1D sequence, TimesNet discovers the
dominant periods of the input via a Fourier transform (FFT) and folds the
sequence into a set of 2D tensors of shape ``[length // period, period]``.
Each period-specific view is passed through an inception-style 2D convolution
block and the per-period results are recombined with a softmax over the
relative period strengths (the ``Topk_Fast`` / ``FFT_for_Period`` mechanism).
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, LayerNormalization

from tfts.contracts import BackboneCapabilities, OutputPort

from .base import BaseModel, CommonConfig
from .registry import register_model


class PositionalEmbedding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # Precompute the [max_len, d_model] sinusoidal table as a numpy buffer.
        # (Keras-3 TF backend: tensors created inside build live in a scratch
        # FuncGraph and can't be read from call, so we keep raw numpy and
        # convert inside call instead.)
        max_len = 5000
        pos = np.arange(max_len, dtype=np.float32)[:, None]
        i = np.arange(d_model, dtype=np.float32)[None, :]
        angle = pos * np.exp(-np.log(10000.0) * (2 * (i // 2)) / d_model)
        pe = np.empty((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = pe

    def call(self, x, **kwargs):
        # slice positions 0..seq_len-1 and broadcast over batch
        seq = tf.shape(x)[-2]
        pe = tf.convert_to_tensor(self.pe)  # fresh constant in the current graph
        pe = tf.gather(pe, tf.range(seq), axis=0)
        return tf.expand_dims(pe, 0)  # [1, seq_len, d_model]


class TokenEmbedding(tf.keras.layers.Layer):
    """Conv1d token embedding with circular padding over time (padding=1, kernel_size=3, padding_mode='circular')."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv1D(filters=d_model, kernel_size=3, padding="valid", use_bias=False)

    def call(self, x, **kwargs):
        # circular padding of 1 on both time ends
        front = x[:, -1:, :]
        back = x[:, :1, :]
        x = tf.concat([front, x, back], axis=1)
        return self.conv(x)


class DataEmbedding(tf.keras.layers.Layer):
    """Token embedding + positional embedding + dropout."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.value = TokenEmbedding(d_model)
        self.position = PositionalEmbedding(d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=None):
        x = self.value(x) + self.position(x)
        return self.dropout(x, training=training)


class InceptionBlock(tf.keras.layers.Layer):
    """``Inception_Block_V1`` from ``layers/Conv_Blocks.py``.

    A set of 2D convolutions with growing square kernels ``(2i+1, 2i+1)`` (all
    with ``same`` padding) whose outputs are averaged. In ordinary TF channels-
    last layout the input is ``[B, H, W, C]`` and the kernels are stacked on a
    new final axis before being mean-reduced.
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.num_kernels = num_kernels
        self.convs = [
            Conv2D(out_channels, (i + i + 1, i + i + 1), padding="same", activation=None) for i in range(num_kernels)
        ]

    def call(self, x, **kwargs):
        res = [conv(x) for conv in self.convs]
        return tf.reduce_mean(tf.stack(res, axis=-1), axis=-1)


class TimesBlock(tf.keras.layers.Layer):
    """One TimesNet block: FFT-based period discovery + per-period 2D conv + recombine."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k = top_k
        if activation == "gelu":
            self.act = tf.nn.gelu
        elif activation == "relu":
            self.act = tf.nn.relu
        else:
            raise ValueError(f"Unsupported activation {activation!r}")

        self.conv = tf.keras.Sequential(
            [
                InceptionBlock(d_model, d_ff, num_kernels=num_kernels),
                tf.keras.layers.Activation(self.act),
                InceptionBlock(d_ff, d_model, num_kernels=num_kernels),
            ]
        )

    def call(self, x, training=None):
        # x: [B, T, N] where N == d_model (embedded channels)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        N = tf.shape(x)[2]

        # ---- FFT period discovery (FFT_for_Period) ----
        # tf.signal.rfft only works on the last axis, so put time there.
        x_t = tf.transpose(x, [0, 2, 1])  # [B, N, T]
        xf = tf.signal.rfft(x_t)  # [B, N, T//2+1]
        amp = tf.abs(xf)  # [B, N, T//2+1]
        amp = tf.transpose(amp, [0, 2, 1])  # [B, T//2+1, N]
        frequency_list = tf.reduce_mean(tf.reduce_mean(amp, axis=0), axis=-1)  # [T//2+1]
        freq_range = tf.range(tf.shape(frequency_list)[0])
        frequency_list = tf.where(tf.equal(freq_range, 0), tf.zeros_like(frequency_list), frequency_list)
        _, top_list = tf.math.top_k(frequency_list, k=self.k)  # top freq indices
        periods = T // top_list  # [k] period lengths

        period_weight = tf.reduce_mean(amp, axis=-1)  # [B, T//2+1]
        period_weight = tf.gather(period_weight, top_list, axis=1)  # [B, k]

        # ---- per-period 2D convolution ----
        res_list = []
        for i in range(self.k):
            period = periods[i]
            # pad the tail so the sequence is a whole number of periods
            length = tf.cast(
                tf.math.ceil(tf.cast(T, tf.float32) / tf.cast(period, tf.float32)) * tf.cast(period, tf.float32),
                tf.int32,
            )
            pad_len = length - T
            out = tf.concat([x, tf.zeros([B, pad_len, N], dtype=x.dtype)], axis=1)  # [B, length, N]
            out = tf.reshape(out, [B, length // period, period, N])  # [B, H, W, N]
            out = self.conv(out)  # [B, H, W, N]
            out = tf.reshape(out, [B, length, N])
            res_list.append(out[:, :T, :])

        res = tf.stack(res_list, axis=-1)  # [B, T, N, k]
        pw = tf.nn.softmax(period_weight, axis=-1)  # [B, k]
        pw = pw[:, None, None, :]  # [B, 1, 1, k]
        res = tf.reduce_sum(res * pw, axis=-1)  # [B, T, N]
        return res + x  # residual


class TimesNetConfig(CommonConfig):
    model_type: str = "timesnet"

    def __init__(
        self,
        hidden_size: int = 32,  # d_model
        intermediate_size: int = 32,  # d_ff
        num_layers: int = 2,  # e_layers
        num_attention_heads: int = 4,  # kept for interface consistency (unused)
        top_k: int = 5,
        num_kernels: int = 6,
        hidden_dropout_prob: float = 0.0,  # dropout
        hidden_act: str = "gelu",  # activation
        layer_norm_eps: float = 1e-5,
        c_out: Optional[int] = None,  # output channels; None -> input channels
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.c_out = c_out
        self.initializer_range = initializer_range
        self.update(kwargs)


@register_model(
    "timesnet",
    config=TimesNetConfig,
    paper="https://arxiv.org/abs/2210.02186",
    tags=("attention", "periodic", "convolutional"),
    capabilities=BackboneCapabilities(
        output_ports=frozenset({OutputPort.TEMPORAL_SEQUENCE, OutputPort.POOLED, OutputPort.NATIVE_FORECAST})
    ),
)
class TimesNet(BaseModel):
    """TensorFlow TimesNet for time series forecasting."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TimesNetConfig] = None):
        config = config or TimesNetConfig()
        super().__init__(predict_sequence_length=predict_sequence_length, config=config)

        cfg = self.config
        self.enc_embedding = DataEmbedding(cfg.hidden_size, cfg.hidden_dropout_prob)
        self.blocks = [
            TimesBlock(
                cfg.hidden_size,
                cfg.intermediate_size,
                top_k=cfg.top_k,
                num_kernels=cfg.num_kernels,
                dropout_rate=cfg.hidden_dropout_prob,
                activation=cfg.hidden_act,
            )
            for _ in range(cfg.num_layers)
        ]
        self.layer_norm = LayerNormalization(epsilon=cfg.layer_norm_eps)
        self.predict_linear = None
        self.projection = None
        self._seq_len = None

    def build(self, input_shape):
        value_shape, encoder_shape = self._input_shapes(input_shape)
        self._seq_len = int(encoder_shape[1])
        c_out = int(self.config.c_out or value_shape[-1])
        self.predict_linear = Dense(self._seq_len + self.predict_sequence_length)
        self.projection = Dense(c_out)
        super().build(input_shape)

    def call(
        self, inputs, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, training=None
    ):
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)

        # ---- instance normalization (Non-stationary Transformer style) ----
        means = tf.stop_gradient(tf.reduce_mean(encoder_feature, axis=1, keepdims=True))
        x_norm = encoder_feature - means
        stdev = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(x_norm), axis=1, keepdims=True) + 1e-5))
        x_norm = x_norm / stdev

        # ---- embedding + temporal upsampling (predict_linear) ----
        enc_out = self.enc_embedding(x_norm, training=training)  # [B, T, d_model]
        T = int(enc_out.shape[1])
        if T != self._seq_len:
            raise ValueError(f"TimesNet was built for sequence length {self._seq_len}, but received {T}.")
        enc_out = tf.transpose(enc_out, [0, 2, 1])  # [B, d_model, T]
        enc_out = self.predict_linear(enc_out)  # [B, d_model, T+pred]
        enc_out = tf.transpose(enc_out, [0, 2, 1])  # [B, T+pred, d_model]

        # ---- stacked TimesBlocks ----
        for block in self.blocks:
            enc_out = self.layer_norm(block(enc_out, training=training))

        if output_hidden_states:
            return enc_out

        # ---- projection + de-normalization ----
        c_out = int(self.config.c_out or x.shape[-1])
        dec_out = self.projection(enc_out)  # [B, T+pred, c_out]
        dec_out = dec_out * stdev[:, :, :c_out] + means[:, :, :c_out]

        return dec_out[:, -self.predict_sequence_length :, :]
