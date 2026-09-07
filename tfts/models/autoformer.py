"""
`Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
<https://arxiv.org/abs/2106.13008>`_

Reference-faithful port of the THUML Time-Series-Library Autoformer (`models/Autoformer.py` +
`layers/Autoformer_EncDec.py`) into the tfts API: series decomposition (moving-average),
Auto-Correlation encoder/decoder with *progressive decomposition* in the residuals, and a
trend + seasonal decoder reconstruction. Auto-Correlation length mismatch (decoder query vs
encoder key) is handled inside `AutoCorrelation` (pad/trim key-values to query length).
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
        hidden_size=64,
        num_layers=2,
        num_decoder_layers=1,
        num_attention_heads=4,
        ffn_intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.05,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        positional_type=None,
        use_cache=True,
        classifier_dropout=0.0,
        label_len=0,
        target_dim=1,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.positional_type = positional_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.label_len = label_len
        self.target_dim = target_dim
        self.update(kwargs)


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
        self.label_len = getattr(c, "label_len", 0)
        d_model = c.hidden_size
        d_ff = c.ffn_intermediate_size
        heads = c.num_attention_heads
        moving_avg = c.kernel_size
        dropout = c.hidden_dropout_prob
        act = tf.nn.gelu if c.hidden_act == "gelu" else tf.nn.relu
        target_dim = getattr(c, "target_dim", 1)
        self.target_dim = target_dim

        self.series_decomp = SeriesDecomp(moving_avg)
        # value embeddings (DataEmbedding_wo_pos, marks absent -> value only)
        self.enc_embedding = Dense(d_model)
        self.dec_embedding = Dense(d_model)
        # encoder
        self.encoder_layers = [
            EncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=heads,
                moving_avg=moving_avg,
                dropout=dropout,
                act=act,
            )
            for _ in range(c.num_layers)
        ]
        self.enc_norm = LayerNormalization(epsilon=c.layer_norm_eps)
        # decoder
        self.decoder_layers = [
            DecoderLayer(
                d_model=d_model,
                c_out=target_dim,
                d_ff=d_ff,
                num_heads=heads,
                moving_avg=moving_avg,
                dropout=dropout,
                act=act,
            )
            for _ in range(c.num_decoder_layers)
        ]
        self.dec_norm = LayerNormalization(epsilon=c.layer_norm_eps)
        self.project = Dense(target_dim, activation=None)

    def call(self, inputs, training=None, output_hidden_states=None, return_dict=None, teacher=None):
        """Forward pass of the AutoFormer model (reference Autoformer.forecast).

        Returns shape (batch_size, predict_sequence_length, 1) for a univariate panel.
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)
        B = tf.shape(encoder_feature)[0]
        pred = self.predict_sequence_length
        label_len = self.label_len
        target = x[..., : self.target_dim]
        tf.debugging.assert_equal(
            tf.shape(target)[-1],
            self.target_dim,
            message="AutoFormer input has fewer channels than target_dim",
        )

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

        # encoder
        enc = self.enc_embedding(encoder_feature)  # (B, seq, d_model)
        for e in self.encoder_layers:
            enc = e(enc, training=training)
        enc = self.enc_norm(enc)

        # decoder
        dec = self.dec_embedding(seasonal_init)  # (B, seq+pred, d_model)
        trend = trend_init  # (B, seq+pred, n_feat)
        for d in self.decoder_layers:
            dec, residual_trend = d(dec, enc, training=training)
            trend = trend + residual_trend
        dec = self.dec_norm(dec)  # seasonal part (B, seq+pred, d_model)
        seasonal_part = self.project(dec)  # (B, seq+pred, n_feat)
        out = seasonal_part + trend
        return out[:, -pred:, :]


class EncoderLayer(tf.keras.layers.Layer):
    """Autoformer encoder layer: Auto-Correlation self-attention + progressive decomp residuals."""

    def __init__(
        self,
        kernel_size=None,
        d_model=None,
        num_attention_heads=None,
        dropout_rate=0.1,
        *,
        d_ff=None,
        num_heads=None,
        moving_avg=None,
        dropout=None,
        act=None,
        **kwargs,
    ):
        # Keep the historical four-argument constructor usable by callers while
        # allowing the reference implementation to provide its full settings by
        # keyword. The legacy layer has a model-sized feed-forward projection.
        legacy = d_ff is None and moving_avg is None and num_heads is None
        if d_model is None:
            raise TypeError("d_model is required")
        num_heads = num_attention_heads if num_heads is None else num_heads
        moving_avg = kernel_size if moving_avg is None else moving_avg
        d_ff = d_model if d_ff is None else d_ff
        dropout = dropout_rate if dropout is None else dropout
        act = tf.nn.relu if act is None else act
        super().__init__(**kwargs)
        self.attention = AutoCorrelation(d_model, num_heads)
        self.dropout = Dropout(dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.conv1 = Dense(d_ff)
        self.conv2 = Dense(d_model)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.act = act
        self._legacy_api = legacy

    def call(self, x, training=False):
        x = x + self.dropout(self.attention(x, x, x), training=training)
        x, _ = self.decomp1(x)
        y = self.dropout(self.act(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        res, _ = self.decomp2(x + y)
        return res


class DecoderLayer(tf.keras.layers.Layer):
    """Autoformer decoder layer: self + cross Auto-Correlation, progressive decomp, trend projection."""

    def __init__(
        self,
        kernel_size=None,
        d_model=None,
        num_attention_heads=None,
        drop_rate=0.1,
        *,
        c_out=None,
        d_ff=None,
        num_heads=None,
        moving_avg=None,
        dropout=None,
        act=None,
        **kwargs,
    ):
        legacy = d_ff is None and moving_avg is None and num_heads is None
        if d_model is None:
            raise TypeError("d_model is required")
        num_heads = num_attention_heads if num_heads is None else num_heads
        moving_avg = kernel_size if moving_avg is None else moving_avg
        d_ff = d_model if d_ff is None else d_ff
        dropout = drop_rate if dropout is None else dropout
        act = tf.nn.relu if act is None else act
        c_out = d_model if c_out is None else c_out
        super().__init__(**kwargs)
        self.self_attention = AutoCorrelation(d_model, num_heads)
        self.cross_attention = AutoCorrelation(d_model, num_heads)
        self.dropout = Dropout(dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.conv1 = Dense(d_ff)
        self.conv2 = Dense(d_model)
        self.trend_proj = Dense(c_out)  # projects residual trend d_model -> c_out
        self.act = act
        self._legacy_api = legacy

    def call(self, x, cross, training=False):
        x = x + self.dropout(self.self_attention(x, x, x), training=training)
        x, t1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross), training=training)
        x, t2 = self.decomp2(x)
        y = self.dropout(self.act(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        x, t3 = self.decomp3(x + y)
        residual_trend = self.trend_proj(t1 + t2 + t3)  # (B, decor_len, c_out)
        return x if self._legacy_api else (x, residual_trend)
