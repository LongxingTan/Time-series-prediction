"""
`TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
<https://arxiv.org/abs/2402.19072>`_


TimeXer builds a cross-way attention between native-global exogenous variables
and the target series. Every variate is first windowed into patches (the
"en" branch); in parallel all variates are embedded across the whole time
axis as variate tokens (the inverted "ex" branch). A global learnable token per
variate attends to the exogenous/inverted embedding and is merged back, then a
FFN produces the forecast.
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from tfts.layers.attention_layer import Attention

from .base import BaseModel, CommonConfig
from .registry import register_model


class PositionalEmbedding(tf.keras.layers.Layer):
    """Fixed sinusoidal positional encoding."""

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
        seq = tf.shape(x)[-2]
        pe = tf.convert_to_tensor(self.pe)  # fresh constant in the current graph
        pe = tf.gather(pe, tf.range(seq), axis=0)
        return tf.expand_dims(pe, 0)  # [1, seq_len, d_model]


class PatchEmbedding(tf.keras.layers.Layer):
    """En-branch: window each variate into non-overlapping patches of
    ``patch_len`` and embed them, prepending a learnable global token.

    ``EnEmbedding`` from the reference. Input ``[B, n_vars, T]``, output
    ``[B * n_vars, patch_num + 1, d_model]``.
    """

    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.value_embedding = Dense(d_model, use_bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = Dropout(dropout_rate)
        self.glb_token = self.add_weight(
            name="glb_token",
            shape=[1, n_vars, 1, d_model],
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x, training=None):
        # x: [B, n_vars, T]
        B = tf.shape(x)[0]
        n_vars = tf.shape(x)[1]
        T = tf.shape(x)[2]
        patch_num = T // self.patch_len

        # window into patches: -> [B, n_vars, patch_num, patch_len]
        x_unfold = tf.reshape(x[:, :, : patch_num * self.patch_len], [B, n_vars, patch_num, self.patch_len])
        x_unfold = tf.reshape(x_unfold, [B * n_vars, patch_num, self.patch_len])

        # input encoding
        x_enc = self.value_embedding(x_unfold) + self.position_embedding(x_unfold)  # [B*N, pn, d]
        x_enc = tf.reshape(x_enc, [B, n_vars, patch_num, x_enc.shape[-1]])

        # cat global token -> [B, n_vars, patch_num + 1, d]
        glb = tf.tile(self.glb_token, [B, 1, 1, 1])
        x_enc = tf.concat([x_enc, glb], axis=2)
        x_enc = tf.reshape(x_enc, [B * n_vars, patch_num + 1, x_enc.shape[-1]])
        return self.dropout(x_enc, training=training), self.n_vars


class InvertedEmbedding(tf.keras.layers.Layer):
    """Ex-branch: embed each variate across the full time axis as a single token.

    ``DataEmbedding_inverted`` from the reference. Input ``[B, T, n_vars]`` was
    already time-normalised; output ``[B, n_vars, d_model]``.
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.value = Dense(d_model, use_bias=True)

    def call(self, x, **kwargs):
        # x: [B, T, n_vars] -> transpose -> [B, n_vars, T] -> Dense(T -> d_model)
        x = tf.transpose(x, [0, 2, 1])
        return self.value(x)


class EncoderLayer(tf.keras.layers.Layer):
    """One TimeXer encoder layer.

    Self-attention over the (per-variate) patch sequence, then cross-attention
    that lets each variate's global token attend to the inverted exogenous
    embedding, followed by a per-variate FFN (kernel-size-1 convolutions).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout_rate: float = 0.1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_vars = None  # filled on first call
        self.self_attention = Attention(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.cross_attention = Attention(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.conv1 = Dense(d_ff)  # per-position linear (1x1 conv over time)
        self.conv2 = Dense(d_model)
        self.norm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.norm2 = LayerNormalization(epsilon=layer_norm_eps)
        self.norm3 = LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = Dropout(dropout_rate)
        self.activation = tf.nn.gelu if hidden_act == "gelu" else tf.nn.relu
        self.n_vars = n_vars

    def call(self, x, cross, training=None):
        # x: [B*n_vars, patch_num+1, d]; cross: [B, n_vars, d]
        B = tf.shape(cross)[0]
        D = tf.shape(x)[-1]
        n_vars = tf.shape(cross)[1]

        x = x + self.dropout(self.self_attention(x, x, x, training=training), training=training)
        x = self.norm1(x)

        # pop the global token of each variate and cross-attend to ex_embed
        x_glb_ori = x[:, -1:, :]  # [B*n_vars, 1, d]
        x_glb = tf.reshape(x_glb_ori, [-1, D])  # [B*n_vars, d]
        x_glb = tf.reshape(x_glb, [B, n_vars, D])  # [B, n_vars, d]
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, cross, training=training), training=training
        )  # [B, n_vars, d]
        x_glb = x_glb_ori + tf.reshape(x_glb_attn, [B * n_vars, 1, D])
        x_glb = self.norm2(x_glb)

        x = tf.concat([x[:, :-1, :], x_glb], axis=1)  # [B*n_vars, patch_num+1, d]

        # FFN via pointwise linear over the embedding dim at each patch position
        y = self.dropout(self.activation(self.conv1(x)), training=training)
        y = self.dropout(self.conv2(y), training=training)
        return self.norm3(x + y)


class TimeXerConfig(CommonConfig):
    model_type: str = "timexer"

    def __init__(
        self,
        hidden_size: int = 16,  # d_model
        intermediate_size: int = 32,  # d_ff
        num_layers: int = 2,  # e_layers
        num_attention_heads: int = 4,  # n_heads
        patch_len: int = 16,
        use_norm: bool = True,
        hidden_dropout_prob: float = 0.1,  # dropout
        attention_probs_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",  # activation
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.patch_len = patch_len
        self.use_norm = use_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.update(kwargs)


class FlattenHead(tf.keras.layers.Layer):
    """ "FlattenHead" of the reference: flatten d_model x patch_num -> predict_len."""

    def __init__(self, n_vars, nf, target_window, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_vars = n_vars
        self.linear = Dense(target_window)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=None):
        # x: [B, n_vars, d_model, patch_num+1]
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, self.n_vars, -1])  # flatten last two dims
        x = self.linear(x)
        x = self.dropout(x, training=training)
        return x  # [B, n_vars, pred_len]


@register_model(
    "timexer",
    config=TimeXerConfig,
    paper="https://arxiv.org/abs/2402.19072",
    tags=("attention", "multivariate", "patch"),
)
class TimeXer(BaseModel):
    """TensorFlow TimeXer for multivariate time series forecasting."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TimeXerConfig] = None):
        config = config or TimeXerConfig()
        super().__init__(predict_sequence_length=predict_sequence_length, config=config)

        # n_vars is filled lazily on first forward (depends on input channels).
        self.en_embedding = None
        self.ex_embedding = None
        self.blocks = None
        self.head = None
        self.n_vars = None

    def build(self, input_shape):
        _, encoder_shape = self._input_shapes(input_shape)
        self.n_vars = int(encoder_shape[-1])
        patch_num = int(encoder_shape[1]) // self.config.patch_len
        self.en_embedding = PatchEmbedding(
            self.n_vars, self.config.hidden_size, self.config.patch_len, self.config.hidden_dropout_prob
        )
        self.ex_embedding = InvertedEmbedding(self.config.hidden_size)
        self.blocks = [
            EncoderLayer(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.hidden_dropout_prob,
                self.config.hidden_act,
                self.config.layer_norm_eps,
            )
            for _ in range(self.config.num_layers)
        ]
        self.head = FlattenHead(
            self.n_vars,
            self.config.hidden_size * (patch_num + 1),
            self.predict_sequence_length,
            self.config.hidden_dropout_prob,
        )
        super().build(input_shape)

    def call(
        self, inputs, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, training=None
    ):
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)

        # ---- instance normalization ----
        if self.config.use_norm:
            means = tf.stop_gradient(tf.reduce_mean(encoder_feature, axis=1, keepdims=True))
            x_norm = encoder_feature - means
            stdev = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(x_norm), axis=1, keepdims=True) + 1e-5))
            x_norm = x_norm / stdev
        else:
            x_norm = encoder_feature
            means = stdev = tf.constant(0.0)

        n_vars = int(encoder_feature.shape[-1])

        if n_vars != self.n_vars:
            raise ValueError(f"TimeXer was built for {self.n_vars} variables, but received {n_vars}.")

        # ---- en branch: patch each variate (features='M' path) ----
        x_patch = tf.transpose(x_norm, [0, 2, 1])  # [B, n_vars, T]
        en_embed, _ = self.en_embedding(x_patch, training=training)  # [B*n_vars, pn+1, d]

        # ---- ex branch: inverted / exogenous embedding ----
        ex_embed = self.ex_embedding(x_norm)  # [B, n_vars, d]

        # ---- stacked encoder layers (global-token cross-attention) ----
        for block in self.blocks:
            en_embed = block(en_embed, ex_embed, training=training)

        # ---- reshape back per variate + head ----
        B = tf.shape(ex_embed)[0]
        pn = int(en_embed.shape[1]) - 1
        D = int(en_embed.shape[-1])
        enc_out = tf.reshape(en_embed, [B, n_vars, pn + 1, D])
        if output_hidden_states:
            return tf.reshape(enc_out, [B, n_vars * (pn + 1), D])
        enc_out = tf.transpose(enc_out, [0, 1, 3, 2])  # [B, n_vars, d, pn+1]
        dec_out = self.head(enc_out, training=training)  # [B, n_vars, pred_len]
        dec_out = tf.transpose(dec_out, [0, 2, 1])  # [B, pred_len, n_vars]

        # ---- de-normalization ----
        if self.config.use_norm:
            dec_out = dec_out * stdev[:, 0:1, :] + means[:, 0:1, :]
        return dec_out
