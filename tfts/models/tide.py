"""
`Long-term Forecasting with TiDE: Time-series Dense Encoder
<https://arxiv.org/pdf/2304.08424v1.pdf>`_

Faithful port of the reference THUML Time-Series-Library TiDE (`models/TiDE.py`) into the tfts API,
so the tfts model reproduces the reference implementation's performance (instance normalization,
flattened residual-MLP encoder/decoder, temporal decoder + linear residual projection).
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from .base import BaseModel, CommonConfig
from .registry import register_model

logger = logging.getLogger(__name__)


class TideConfig(CommonConfig):
    model_type: str = "tide"

    def __init__(
        self,
        hidden_size: int = 64,  # reference d_model (ResBlock hidden)
        num_layers: int = 2,  # reference e_layers (encoder ResBlocks)
        decoder_layers: int = 1,  # reference d_layers
        ffn_intermediate_size: int = 128,  # reference d_ff (temporal decoder hidden)
        hidden_dropout_prob: float = 0.05,
        layer_norm_eps: float = 1e-5,
        feature_encode_dim: int = 2,
        feature_dim: int = 4,  # temporal-mark width the reference uses for freq='h'
        target_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_layers = decoder_layers
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.feature_encode_dim = feature_encode_dim
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.update(kwargs)


@register_model(
    "tide", config=TideConfig, paper="https://arxiv.org/abs/2304.08424", tags=("mlp", "efficient", "covariates")
)
class Tide(BaseModel):
    """TiDE model for time series forecasting (reference-portable dense-encoder MLP)."""

    def __init__(self, predict_sequence_length=1, config: Optional[TideConfig] = None):
        super(Tide, self).__init__()
        self.config = config or TideConfig()
        self.predict_sequence_length = predict_sequence_length

        # Resolve fields with defaults so both a hand-built TideConfig and the AutoConfig
        # path (which only carries the keys passed in) work identically.
        c = self.config
        hidden = getattr(c, "hidden_size", 64)
        num_layers = getattr(c, "num_layers", 2)
        decoder_layers = getattr(c, "decoder_layers", 1)
        ffn = getattr(c, "ffn_intermediate_size", 128)
        dropout = getattr(c, "hidden_dropout_prob", 0.05)
        eps = getattr(c, "layer_norm_eps", 1e-5)
        self.feature_encode_dim = getattr(c, "feature_encode_dim", 2)
        self.feature_dim = getattr(c, "feature_dim", 4)
        self.target_dim = getattr(c, "target_dim", 1)

        self.feature_encoder = ResBlock(hidden, self.feature_encode_dim, dropout, eps)
        self.encoders = [ResBlock(hidden, hidden, dropout, eps) for _ in range(num_layers)]
        self.decoders = [ResBlock(hidden, hidden, dropout, eps) for _ in range(max(0, decoder_layers - 1))]
        self.decoders.append(ResBlock(hidden, 1 * self.predict_sequence_length, dropout, eps))
        self.temporal_decoder = ResBlock(1 + self.feature_encode_dim, ffn, dropout, eps)
        self.residual_proj = Dense(self.predict_sequence_length)

    def call(self, x, training=None, **kwargs):
        """Process per-series windows through the reference TiDE dense-encoder MLP.

        Args:
            x: input of shape (batch_size, seq_len, features).
            training: passed to Dropout/Jupyter inside the blocks.

        Returns:
            tf.Tensor of shape (batch_size, predict_sequence_length, features)
        """
        x, encoder_feature, _ = self._prepare_3d_inputs(x, ignore_decoder_inputs=True)
        seq = tf.shape(encoder_feature)[1]
        batch = tf.shape(encoder_feature)[0]
        target = x[..., : self.target_dim]
        tf.debugging.assert_equal(
            tf.shape(target)[-1],
            self.target_dim,
            message="TiDE input has fewer channels than target_dim",
        )
        n_feat = self.target_dim

        # Instance normalization (RevIN) over the seq dim, exactly like the reference.
        mean = tf.reduce_mean(target, axis=1, keepdims=True)  # (b,1,target_dim)
        stdev = tf.sqrt(tf.reduce_mean(tf.square(target - mean), axis=1, keepdims=True) + 1e-5)
        xn = (target - mean) / stdev  # (b,seq,target_dim)

        # Temporal-mark feature path; marks are absent (zeros) for the covariate-free benchmark.
        marks = tf.zeros([batch, seq + self.predict_sequence_length, self.feature_dim])
        feature = self.feature_encoder(marks)  # (b, seq+pred, feat_encode)
        feature_flat = tf.reshape(feature, [batch, (seq + self.predict_sequence_length) * self.feature_encode_dim])

        outs = []
        for feat in range(n_feat):
            xf = xn[..., feat : feat + 1]  # (b,seq,1)
            xf_flat = tf.reshape(xf, [batch, seq])
            hidden_in = tf.concat([xf_flat, feature_flat], axis=-1)  # (b, seq + feat_width)
            h = hidden_in
            for enc in self.encoders:
                h = enc(h, training=training)
            for dec in self.decoders:
                h = dec(h, training=training)
            decoded = tf.reshape(h, [batch, self.predict_sequence_length, 1])  # (b,pred,1)
            td_in = tf.concat([feature[:, seq:], decoded], axis=-1)  # (b,pred,feat_encode+1)
            dec_out = self.temporal_decoder(td_in, training=training)[..., 0]  # (b,pred)
            out = dec_out + self.residual_proj(xf_flat)  # (b,pred) + Linear(seq->pred)
            # De-normalize
            out = out * tf.squeeze(stdev[..., feat : feat + 1], axis=1) + tf.squeeze(mean[..., feat : feat + 1], axis=1)
            outs.append(out[..., None])
        return tf.concat(outs, axis=-1)


class ResBlock(tf.keras.layers.Layer):
    """Reference TiDE residual MLP block: relu(Linear->hidden) -> Linear->out, residual skip, LayerNorm."""

    def __init__(self, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(output_dim)
        self.fc3 = Dense(output_dim)  # input -> output residual adaptation
        self.ln = LayerNormalization(epsilon=eps)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        out = tf.nn.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.dropout(out, training=training)
        return self.ln(out + self.fc3(x))
