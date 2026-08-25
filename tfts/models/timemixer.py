"""
`TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
<https://arxiv.org/abs/2405.14616>`_

Key idea: TimeMixer builds a multi-scale representation by down-sampling the
input over time, then in each ``PastDecomposableMixing`` block decomposes every
scale into a seasonal and a trend component. The seasonal components are mixed
bottom-up across scales while the trend components are mixed top-down, and the
final prediction is obtained by a multi-scale aggregation (a sum over the
per-scale forecasts produced by a shared ``projection`` and residual
``regression_layers``).

This port targets ``channel_independence=False`` (the global / multivariate
path): the encoder embeds the full set of input channels and projects them back
to ``c_out`` channels, matching tfts' convention of predicting every input
channel at once (``[B, pred_len, n_vars]``).
"""

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LayerNormalization

from .base import BaseConfig, BaseModel


def _clamp_odd_kernel(kernel: int, length: int) -> int:
    """Return an effective (odd) moving-average kernel that fits the sequence.

    A kernel wider than the sequence breaks the pooling, and even kernels are
    not supported by the reference's odd-only symmetric padding, so we coerce
    the requested ``kernel`` to the largest odd value <= ``length``.
    """
    if length < 2:
        return 1
    k = min(kernel, length)
    if k % 2 == 0:
        k -= 1
    return max(1, k)


def _moving_average(x: tf.Tensor, kernel: int) -> tf.Tensor:
    """Edge-padded, per-channel moving average along the time axis (axis 1).

    Uses a cumulative-sum sliding window so it works for any kernel, including kernels larger than the sequence.
    """
    T = int(x.shape[1])
    k = _clamp_odd_kernel(kernel, T)
    if k <= 1:
        return x
    pad = (k - 1) // 2
    front = tf.tile(x[:, :1, :], [1, pad, 1])
    end = tf.tile(x[:, -1:, :], [1, pad, 1])
    xp = tf.concat([front, x, end], axis=1)  # [B, T + k - 1, N]
    cum = tf.cumsum(xp, axis=1)
    zeros = tf.zeros_like(cum[:, :1, :])
    # windowed sums S[t] = sum(xp[t .. t+k-1]), for t in [0, T)
    sums = cum[:, k - 1 :, :] - tf.concat([zeros, cum[:, :-k, :]], axis=1)
    return sums / tf.cast(k, x.dtype)


def _series_decomp(x: tf.Tensor, kernel: int) -> tf.Tensor:
    """Return the seasonal residual (``x - trend``) of a moving-average decomp."""
    trend = _moving_average(x, kernel)
    return x - trend


class DFT_series_decomp(tf.keras.layers.Layer):
    """``DFT_series_decomp`` from the reference (frequency-domain decomposition).

    Keeps only the top-``k`` frequency bins of each channel and reconstructs
    the seasonal component via an inverse real FFT; the trend is the residual.
    """

    def __init__(self, top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def call(self, x, **kwargs):
        # x: [B, T, N] -> FFT over time (last axis for tf.signal.rfft)
        xf = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))  # [B, N, T//2+1]
        xf = tf.transpose(xf, [0, 2, 1])  # [B, T//2+1, N]
        freq = tf.abs(xf)
        freq = tf.concat([tf.zeros_like(freq[:, :1, :]), freq[:, 1:, :]], axis=1)  # zero the DC bin
        k = min(self.top_k, int(freq.shape[1]))
        top_k_freq, _ = tf.math.top_k(freq, k=k, sorted=True)
        threshold = tf.reduce_min(top_k_freq, axis=1, keepdims=True)
        xf = tf.where(freq <= threshold, tf.zeros_like(xf), xf)
        x_season = tf.signal.irfft(tf.transpose(xf, [0, 2, 1]), fft_length=[2 * (int(x.shape[1]) // 2 + 1)])
        x_season = tf.transpose(x_season, [0, 2, 1])[:, : tf.shape(x)[1], :]
        return x_season, x - x_season


class TokenEmbedding(tf.keras.layers.Layer):
    """``TokenEmbedding`` of ``DataEmbedding_wo_pos``: 1d conv with circular pad."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.conv = Conv1D(filters=d_model, kernel_size=3, padding="valid", use_bias=False)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=None):
        front = x[:, -1:, :]
        back = x[:, :1, :]
        x = self.conv(tf.concat([front, x, back], axis=1))
        return self.dropout(x, training=training)


class DataEmbeddingWoPos(tf.keras.layers.Layer):
    """``DataEmbedding_wo_pos``: token embedding + dropout (no positional)."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.value = TokenEmbedding(d_model, dropout_rate)

    def call(self, x, training=None):
        return self.value(x, training=training)


class MultiScaleSeasonMixing(tf.keras.layers.Layer):
    """Bottom-up mixing of the per-scale seasonal components."""

    def build(self, input_shape):
        shapes = input_shape if isinstance(input_shape, (list, tuple)) else [input_shape]
        lengths = [int(s[-1]) for s in shapes]  # [B, d, T_i] -> T_i (time) is last
        self.downs = []
        for i in range(len(lengths) - 1):
            lo = lengths[i + 1]
            self.downs.append(Dense(lo))
            self.downs.append(Dense(lo))
        super().build(input_shape)

    def call(self, season_list, **kwargs):
        # season_list entries are [B, d, T_i]
        out_high = season_list[0]
        out_list = [tf.transpose(out_high, [0, 2, 1])]  # [B, T0, d]
        for i in range(len(season_list) - 1):
            a = self.downs[2 * i]
            b = self.downs[2 * i + 1]
            res = b(tf.nn.gelu(a(out_high)))
            out_low = season_list[i + 1] + res
            out_high = out_low
            out_list.append(tf.transpose(out_high, [0, 2, 1]))
        return out_list


class MultiScaleTrendMixing(tf.keras.layers.Layer):
    """Top-down mixing of the per-scale trend components."""

    def build(self, input_shape):
        shapes = input_shape if isinstance(input_shape, (list, tuple)) else [input_shape]
        lengths = [int(s[-1]) for s in shapes]  # [B, d, T_i] -> time is last
        self.ups = []
        # up[i]: maps length L_{m-1-i} -> L_{m-2-i}
        for i in range(len(lengths) - 1):
            hi = lengths[len(lengths) - 2 - i]
            self.ups.append(Dense(hi))
            self.ups.append(Dense(hi))
        super().build(input_shape)

    def call(self, trend_list, **kwargs):
        rev = trend_list[::-1]  # lowest scale first
        out_low = rev[0]
        out_high = rev[1]
        out_list = [tf.transpose(out_low, [0, 2, 1])]  # [B, T_{m-1}, d]
        for i in range(len(rev) - 1):
            a = self.ups[2 * i]
            b = self.ups[2 * i + 1]
            res = b(tf.nn.gelu(a(out_low)))
            out_high = out_high + res
            out_low = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            out_list.append(tf.transpose(out_low, [0, 2, 1]))
        out_list.reverse()
        return out_list


class PastDecomposableMixing(tf.keras.layers.Layer):
    """A single ``PastDecomposableMixing`` encoder block.

    Decomposes each scale into season + trend, applies a small pointwise MLP
    (``cross_layer``) to both, then re-combines them through the multi-scale
    season (bottom-up) and trend (top-down) mixing.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        moving_avg: int = 25,
        top_k: int = 5,
        decomp_method: str = "moving_avg",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.decomp_method = decomp_method
        self.layer_norm = LayerNormalization(epsilon=1e-5)
        self.dropout = Dropout(0.0)
        # cross layer (season/trend refinement), d_model -> d_ff -> d_model
        self.cross_dense1 = Dense(d_ff)
        self.cross_dense2 = Dense(d_model)
        self.out_dense1 = Dense(d_ff)
        self.out_dense2 = Dense(d_model)
        self.mixing_season = MultiScaleSeasonMixing()
        self.mixing_trend = MultiScaleTrendMixing()
        self._decomp = DFT_series_decomp(self.top_k)

    def _decompose(self, x: tf.Tensor) -> tf.Tensor:
        if self.decomp_method == "moving_avg":
            return _series_decomp(x, self.moving_avg)
        if self.decomp_method == "dft_decomp":
            return self._decomp(x)[0]
        raise ValueError(f"decomp_method '{self.decomp_method}' is not supported")

    def call(self, x_list, training=None):
        length_list = [int(x.shape[1]) for x in x_list]
        season_list = []
        for x in x_list:
            season = self._decompose(x)  # [B, T_i, d_model] seasonal residual
            trend = x - season
            season = tf.nn.gelu(self.cross_dense2(tf.nn.gelu(self.cross_dense1(season))))
            trend = tf.nn.gelu(self.cross_dense2(tf.nn.gelu(self.cross_dense1(trend))))
            season_list.append(tf.transpose(season, [0, 2, 1]))

        trend_list = [
            tf.transpose(
                tf.nn.gelu(self.cross_dense2(tf.nn.gelu(self.cross_dense1(self._decompose(xt))))),
                [0, 2, 1],
            )
            for xt in x_list
        ]

        out_season_list = self.mixing_season(season_list)
        out_trend_list = self.mixing_trend(trend_list)

        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend  # [B, T_i, d_model]
            out = self.layer_norm(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixerConfig(BaseConfig):
    model_type: str = "timemixer"

    def __init__(
        self,
        d_model: int = 32,  # hidden
        d_ff: int = 32,  # intermediate
        e_layers: int = 4,  # number of PDM blocks
        dropout: float = 0.0,
        moving_avg: int = 25,  # moving-average kernel for the decomposition
        down_sampling_window: int = 2,
        down_sampling_layers: int = 1,
        down_sampling_method: str = "avg",  # 'avg' | 'max' | 'conv' | None
        decomp_method: str = "moving_avg",  # 'moving_avg' | 'dft_decomp'
        top_k: int = 5,
        channel_independence: bool = False,
        use_norm: bool = True,
        c_out: Optional[int] = None,  # output channels; None -> input channels
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.dropout = dropout
        self.moving_avg = moving_avg
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.top_k = top_k
        self.channel_independence = channel_independence
        self.use_norm = use_norm
        self.c_out = c_out
        self.update(kwargs)


class TimeMixer(BaseModel):
    """TensorFlow TimeMixer for multivariate time series forecasting.

    The model embeds a multi-scale view of the input, refines it with
    ``e_layers`` ``PastDecomposableMixing`` blocks, and aggregates per-scale
    projections into a single forecast of shape ``[B, pred_len, n_vars]``.
    """

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TimeMixerConfig] = None):
        config = config or TimeMixerConfig()
        super().__init__(predict_sequence_length=predict_sequence_length, config=config)
        self.cfg = self.config
        self.predict_layers = None
        self.projection_layer = None
        # Built lazily on the first forward once n_vars is observed.
        self.embeddings = None
        self.pdm_blocks = None
        self._built = None

    def build(self, input_shape):
        value_shape, encoder_shape = self._input_shapes(input_shape)
        self._build(int(encoder_shape[1]), int(value_shape[-1]))
        super().build(input_shape)

    # ------------------------------------------------------------------ helpers
    def _multi_scale_downsample(self, x: tf.Tensor) -> List[tf.Tensor]:
        """Return ``down_sampling_layers + 1`` scales of ``x`` over time.

        tfts inputs are ``[B, T, N]``; ``tf.nn.avg/max_pool1d`` pool over the
        middle (time) axis and preserve the channel axis, so we pool ``x`` as
        given (no transposition).
        """
        method = self.cfg.down_sampling_method
        if method is None:
            return [x]
        window = self.cfg.down_sampling_window
        n_layers = self.cfg.down_sampling_layers
        out = [x]
        cur = x  # [B, T, N]
        for _ in range(n_layers):
            if method == "avg":
                cur = tf.nn.avg_pool1d(cur, ksize=window, strides=window, padding="VALID")
            elif method == "max":
                cur = tf.nn.max_pool1d(cur, ksize=window, strides=window, padding="VALID")
            else:
                raise ValueError(f"down_sampling_method '{method}' is not supported")
            out.append(cur)
        return out

    def _build(self, T0: int, N: int):
        window = self.cfg.down_sampling_window
        n_scales = self.cfg.down_sampling_layers + 1
        pred = self.predict_sequence_length
        lengths = [T0 // (window**i) for i in range(n_scales)]

        self.predict_layers = [Dense(pred) for _ in lengths]
        self.out_res_layers = [Dense(Ti) for Ti in lengths]
        self.regression_layers = [Dense(pred) for _ in lengths]
        if self.cfg.channel_independence:
            c_out = self.cfg.c_out or 1
        else:
            c_out = int(self.cfg.c_out or N)
        self.projection_layer = Dense(c_out)
        self.embeddings = DataEmbeddingWoPos(self.cfg.d_model, self.cfg.dropout)
        self.pdm_blocks = [
            PastDecomposableMixing(
                self.cfg.d_model,
                self.cfg.d_ff,
                moving_avg=self.cfg.moving_avg,
                top_k=self.cfg.top_k,
                decomp_method=self.cfg.decomp_method,
            )
            for _ in range(self.cfg.e_layers)
        ]
        self._built = (T0, N)

    def call(
        self, inputs, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, training=None
    ):
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)
        N = int(x.shape[-1])

        # ---- multi-scale input + per-scale normalization ----
        x_scales = self._multi_scale_downsample(encoder_feature)
        T0 = int(x_scales[0].shape[1])
        if self._built != (T0, N):
            raise ValueError(f"TimeMixer was built for input shape {self._built}, but received {(T0, N)}.")

        x_list = []
        for xs in x_scales:
            if self.cfg.use_norm:
                mean = tf.stop_gradient(tf.reduce_mean(xs, axis=1, keepdims=True))
                stdev = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(xs - mean), axis=1, keepdims=True) + 1e-5))
                xs = (xs - mean) / stdev
            x_list.append(xs)
        mean0 = tf.stop_gradient(tf.reduce_mean(x_scales[0], axis=1, keepdims=True))
        stdev0 = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(x_scales[0] - mean0), axis=1, keepdims=True) + 1e-5))

        # ---- pre_enc: decompose each scale (global path: season embeds) ----
        season_list = [_series_decomp(xs, self.cfg.moving_avg) for xs in x_list]
        trend_list = [xs - s for xs, s in zip(x_list, season_list)]

        # ---- embedding ----
        enc_out_list = [self.embeddings(s, training=training) for s in season_list]

        # ---- Past Decomposable Mixing encoder, e_layers times ----
        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list, training=training)

        if output_hidden_states:
            return enc_out_list[0]

        # ---- future multi-scale mixing (decoder) ----
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # align temporal dimension: [B, d, T_i] -> [B, d, pred]
            dec_out = tf.transpose(enc_out, [0, 2, 1])
            dec_out = self.predict_layers[i](dec_out)  # [B, d, pred]
            dec_out = tf.transpose(dec_out, [0, 2, 1])  # [B, pred, d]
            dec_out = self.projection_layer(dec_out)  # [B, pred, c_out]
            # residual branch from the pre-encoder trend
            out_res = tf.transpose(trend_list[i], [0, 2, 1])  # [B, N, T_i]
            out_res = self.out_res_layers[i](out_res)  # [B, N, T_i]
            out_res = tf.transpose(self.regression_layers[i](out_res), [0, 2, 1])  # [B, pred, N]
            dec_out = dec_out + out_res
            dec_out_list.append(dec_out)

        dec_out = tf.stack(dec_out_list, axis=-1)  # [B, pred, c_out, n_scales]
        dec_out = tf.reduce_sum(dec_out, axis=-1)  # [B, pred, c_out]

        # ---- denormalize with the scale-0 statistics ----
        if self.cfg.use_norm:
            dec_out = dec_out * stdev0[:, :, :] + mean0[:, :, :]
        return dec_out
