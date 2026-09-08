"""
`Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
<https://arxiv.org/abs/2106.13008>`_
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from tfts.layers.autoformer_layer import AutoCorrelation, SeriesDecomp

from .base import BaseModel, CommonConfig
from .registry import register_model


class AutoFormerConfig(CommonConfig):
    """Configuration to store the config of a [`AutoFormer`]."""

    model_type: str = "autoformer"

    def __init__(
        self,
        kernel_size=7,
        num_decoder_layers=1,
        ffn_intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.05,
        label_len=0,
        autocorrelation_factor=None,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_decoder_layers = num_decoder_layers
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.label_len = label_len
        self.autocorrelation_factor = autocorrelation_factor
        self.update(kwargs)
        if self.num_decoder_layers is None:
            self.num_decoder_layers = self.num_layers


@register_model(
    "autoformer",
    config=AutoFormerConfig,
    paper="https://arxiv.org/abs/2106.13008",
    tags=("decomposition", "seasonal", "SOTA"),
    tier="core",
)
class AutoFormer(BaseModel):
    """AutoFormer model with progressive decomposition and trend+seasonal decoder (reference port)."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[AutoFormerConfig] = None) -> None:
        super().__init__()
        c = config or AutoFormerConfig()
        self.config = c
        self.predict_sequence_length = predict_sequence_length
        self.label_len = c.label_len
        d_model = c.hidden_size
        d_ff = c.ffn_intermediate_size
        heads = c.num_attention_heads
        moving_avg = c.kernel_size
        dropout = c.hidden_dropout_prob
        act = tf.keras.activations.get(c.hidden_act)
        target_dim = c.target_dim
        self.target_dim = target_dim

        self.series_decomp = SeriesDecomp(moving_avg)
        # value embeddings (DataEmbedding_wo_pos, marks absent -> value only)
        self.enc_embedding = Dense(d_model)
        self.dec_embedding = Dense(d_model)
        self.encoder = Encoder(
            num_layers=c.num_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=act,
            factor=c.autocorrelation_factor,
            epsilon=c.layer_norm_eps,
        )
        self.decoder = Decoder(
            num_layers=c.num_decoder_layers,
            d_model=d_model,
            c_out=target_dim,
            d_ff=d_ff,
            num_heads=heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=act,
            factor=c.autocorrelation_factor,
            epsilon=c.layer_norm_eps,
        )

    def build(self, input_shape):
        self._validate_target_shape(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, output_hidden_states=None, return_dict=None, teacher=None):
        """Forward pass of the AutoFormer model (reference Autoformer.forecast).

        Returns shape (batch_size, predict_sequence_length, 1) for a univariate panel.
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)
        B = tf.shape(encoder_feature)[0]
        pred = self.predict_sequence_length
        label_len = self.label_len
        target, _ = self._split_targets(x)

        # decomp init (moving-average decomposition)
        seasonal_init, trend_init = self.series_decomp(target)  # each (B, seq, target_dim)
        mean = tf.reduce_mean(target, axis=1, keepdims=True)  # (B,1,target_dim)
        zeros = tf.zeros([B, pred, self.target_dim], dtype=target.dtype)

        # decoder input: last `label_len` seasonal/trend context + blank mean/zeros
        if label_len > 0:
            trend_ctx = trend_init[:, -label_len:]
            seasonal_ctx = seasonal_init[:, -label_len:]
        else:
            trend_ctx = trend_init
            seasonal_ctx = seasonal_init
        trend_init = tf.concat([trend_ctx, tf.tile(mean, [1, pred, 1])], axis=1)  # (B, seq+pred, n_feat)
        seasonal_init = tf.concat([seasonal_ctx, zeros], axis=1)  # (B, seq+pred, n_feat)

        enc = self.encoder(self.enc_embedding(encoder_feature), training=training)
        out = self.decoder(self.dec_embedding(seasonal_init), enc, trend_init, training=training)
        return out[:, -pred:, :]


@tf.keras.utils.register_keras_serializable(package="tfts")
class EncoderLayer(tf.keras.layers.Layer):
    """Autoformer encoder layer: Auto-Correlation self-attention + progressive decomp residuals."""

    def __init__(self, d_model, d_ff, num_heads, moving_avg, dropout=0.1, act="relu", factor=None, **kwargs):
        super().__init__(**kwargs)
        self.settings = dict(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=tf.keras.activations.serialize(tf.keras.activations.get(act)),
            factor=factor,
        )
        self.attention = AutoCorrelation(d_model, num_heads, factor=factor)
        self.dropout = Dropout(dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.conv1 = Dense(d_ff)
        self.conv2 = Dense(d_model)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.act = tf.keras.activations.get(act)

    def call(self, x, training=False):
        x = x + self.dropout(self.attention(x, x, x), training=training)
        x, _ = self.decomp1(x)
        y = self.dropout(self.act(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        res, _ = self.decomp2(x + y)
        return res

    def get_config(self):
        return dict(super().get_config(), **self.settings)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


@tf.keras.utils.register_keras_serializable(package="tfts")
class DecoderLayer(tf.keras.layers.Layer):
    """Autoformer decoder layer: self + cross Auto-Correlation, progressive decomp, trend projection."""

    def __init__(self, d_model, c_out, d_ff, num_heads, moving_avg, dropout=0.1, act="relu", factor=None, **kwargs):
        super().__init__(**kwargs)
        self.settings = dict(
            d_model=d_model,
            c_out=c_out,
            d_ff=d_ff,
            num_heads=num_heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=tf.keras.activations.serialize(tf.keras.activations.get(act)),
            factor=factor,
        )
        self.self_attention = AutoCorrelation(d_model, num_heads, factor=factor)
        self.cross_attention = AutoCorrelation(d_model, num_heads, factor=factor)
        self.dropout = Dropout(dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.conv1 = Dense(d_ff)
        self.conv2 = Dense(d_model)
        self.trend_proj = Dense(c_out)  # projects residual trend d_model -> c_out
        self.act = tf.keras.activations.get(act)

    def call(self, x, cross, training=False):
        x = x + self.dropout(self.self_attention(x, x, x), training=training)
        x, t1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross), training=training)
        x, t2 = self.decomp2(x)
        y = self.dropout(self.act(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        x, t3 = self.decomp3(x + y)
        residual_trend = self.trend_proj(t1 + t2 + t3)  # (B, decor_len, c_out)
        return x, residual_trend

    def get_config(self):
        return dict(super().get_config(), **self.settings)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape, shape[:-1].concatenate(self.settings["c_out"])


@tf.keras.utils.register_keras_serializable(package="tfts")
class Encoder(tf.keras.layers.Layer):
    """Stack of progressive decomposition encoder layers."""

    def __init__(
        self,
        num_layers,
        d_model,
        d_ff,
        num_heads,
        moving_avg,
        dropout=0.1,
        act="relu",
        factor=None,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        settings = dict(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=tf.keras.activations.serialize(tf.keras.activations.get(act)),
            factor=factor,
        )
        self.settings = dict(settings, num_layers=num_layers, epsilon=epsilon)
        self.layers = [EncoderLayer(**settings) for _ in range(num_layers)]
        self.norm = LayerNormalization(epsilon=epsilon)

    def call(self, x, training=None):
        for layer in self.layers:
            x = layer(x, training=training)
        return self.norm(x)

    def get_config(self):
        settings = dict(self.settings)
        settings["act"] = tf.keras.activations.serialize(tf.keras.activations.get(settings["act"]))
        return dict(super().get_config(), **settings)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


@tf.keras.utils.register_keras_serializable(package="tfts")
class Decoder(tf.keras.layers.Layer):
    """Seasonal decoder with accumulated trend reconstruction."""

    def __init__(
        self,
        num_layers,
        d_model,
        c_out,
        d_ff,
        num_heads,
        moving_avg,
        dropout=0.1,
        act="relu",
        factor=None,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        settings = dict(
            c_out=c_out,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            moving_avg=moving_avg,
            dropout=dropout,
            act=tf.keras.activations.serialize(tf.keras.activations.get(act)),
            factor=factor,
        )
        self.settings = dict(settings, num_layers=num_layers, epsilon=epsilon)
        self.layers = [DecoderLayer(**settings) for _ in range(num_layers)]
        self.norm = LayerNormalization(epsilon=epsilon)
        self.projection = Dense(settings["c_out"])

    def call(self, x, cross, trend, training=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, training=training)
            trend = trend + residual_trend
        return self.projection(self.norm(x)) + trend

    def get_config(self):
        settings = dict(self.settings)
        settings["act"] = tf.keras.activations.serialize(tf.keras.activations.get(settings["act"]))
        return dict(super().get_config(), **settings)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)[:-1].concatenate(self.settings["c_out"])
