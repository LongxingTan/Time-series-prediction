"""
`Long-term Forecasting with TiDE: Time-series Dense Encoder
<https://arxiv.org/pdf/2304.08424v1.pdf>`_

Dense encoder/decoder with reversible instance normalization, an optional shared
history/future covariate encoder, and a temporal decoder plus linear residual.
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from tfts.layers.revin import RevIN

from .base import BaseModel, CommonConfig
from .registry import register_model

logger = logging.getLogger(__name__)


class TideConfig(CommonConfig):
    model_type: str = "tide"

    def __init__(
        self,
        decoder_layers: int = 1,  # reference d_layers
        ffn_intermediate_size: int = 128,  # reference d_ff (temporal decoder hidden)
        hidden_dropout_prob: float = 0.05,
        feature_encode_dim: int = 2,
        feature_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.ffn_intermediate_size = ffn_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.feature_encode_dim = feature_encode_dim
        self.feature_dim = feature_dim
        self.update(kwargs)

    def __post_init__(self):
        if self.feature_dim < 0 or self.feature_encode_dim < 1 or self.decoder_layers < 1:
            raise ValueError("feature_dim must be nonnegative; feature_encode_dim and decoder_layers must be positive")


@register_model(
    "tide", config=TideConfig, paper="https://arxiv.org/abs/2304.08424", tags=("mlp", "efficient", "covariates")
)
class Tide(BaseModel):
    """TiDE model for time series forecasting (reference-portable dense-encoder MLP)."""

    def __init__(self, predict_sequence_length=1, config: Optional[TideConfig] = None):
        super(Tide, self).__init__()
        self.config = config or TideConfig()
        self.predict_sequence_length = predict_sequence_length

        c = self.config
        hidden = c.hidden_size
        num_layers = c.num_layers
        decoder_layers = c.decoder_layers
        ffn = c.ffn_intermediate_size
        dropout = c.hidden_dropout_prob
        eps = c.layer_norm_eps
        self.feature_encode_dim = c.feature_encode_dim
        self.feature_dim = c.feature_dim
        self.target_dim = c.target_dim
        self.revin = RevIN()
        self.feature_encoder = ResBlock(hidden, self.feature_encode_dim, dropout, eps) if self.feature_dim else None
        self.encoders = [ResBlock(hidden, hidden, dropout, eps) for _ in range(num_layers)]
        self.decoders = [ResBlock(hidden, hidden, dropout, eps) for _ in range(max(0, decoder_layers - 1))]
        self.decoders.append(ResBlock(hidden, 1 * self.predict_sequence_length, dropout, eps))
        self.temporal_decoder = ResBlock(ffn, 1, dropout, eps, normalize=False)
        self.residual_proj = Dense(self.predict_sequence_length)

    def build(self, input_shape):
        self._validate_target_shape(input_shape)
        if self.feature_dim:
            if isinstance(input_shape, dict):
                history_shape = input_shape["encoder_feature"]
                future_shape = input_shape.get("decoder_feature")
            elif isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
                history_shape, future_shape = input_shape[1:]
            else:
                raise ValueError("TiDE feature_dim > 0 requires encoder_feature and decoder_feature")
            if future_shape is None or history_shape[-1] != self.feature_dim or future_shape[-1] != self.feature_dim:
                raise ValueError("TiDE encoder_feature and decoder_feature widths must match feature_dim")
            if future_shape[1] is not None and future_shape[1] != self.predict_sequence_length:
                raise ValueError("TiDE decoder_feature length must match the prediction horizon")
        super().build(input_shape)

    def call(self, x, training=None, **kwargs):
        """Process per-series windows through the reference TiDE dense-encoder MLP.

        Args:
            x: input of shape (batch_size, seq_len, features).
            training: controls dropout inside the residual blocks.

        Returns:
            tf.Tensor of shape (batch_size, predict_sequence_length, features)
        """
        inputs = x
        x, encoder_feature, _ = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=True)
        decoder_feature = (
            inputs.get("decoder_feature")
            if isinstance(inputs, dict)
            else (inputs[2] if isinstance(inputs, (tuple, list)) else None)
        )
        seq = tf.shape(encoder_feature)[1]
        batch = tf.shape(encoder_feature)[0]
        target, _ = self._split_targets(x)
        n_feat = self.target_dim

        xn, stats = self.revin(target)
        feature = None
        if self.feature_encoder is not None:
            if decoder_feature is None:
                raise ValueError("TiDE feature_dim > 0 requires encoder_feature and decoder_feature")
            history_marks = encoder_feature[..., -self.feature_dim :]
            marks = tf.concat([history_marks, decoder_feature], axis=1)
            feature = self.feature_encoder(marks, training=training)
            feature_flat = tf.reshape(feature, [batch, (seq + self.predict_sequence_length) * self.feature_encode_dim])

        outs = []
        for feat in range(n_feat):
            xf = xn[..., feat : feat + 1]  # (b,seq,1)
            xf_flat = tf.reshape(xf, [batch, seq])
            hidden_in = xf_flat if feature is None else tf.concat([xf_flat, feature_flat], axis=-1)
            h = hidden_in
            for enc in self.encoders:
                h = enc(h, training=training)
            for dec in self.decoders:
                h = dec(h, training=training)
            decoded = tf.reshape(h, [batch, self.predict_sequence_length, 1])  # (b,pred,1)
            td_in = decoded if feature is None else tf.concat([feature[:, seq:], decoded], axis=-1)
            dec_out = self.temporal_decoder(td_in, training=training)[..., 0]  # (b,pred)
            out = dec_out + self.residual_proj(xf_flat)  # (b,pred) + Linear(seq->pred)
            outs.append(out[..., None])
        return self.revin.inverse(tf.concat(outs, axis=-1), stats)


class ResBlock(tf.keras.layers.Layer):
    """Reference TiDE residual MLP block: relu(Linear->hidden) -> Linear->out, residual skip, LayerNorm."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        eps: float = 1e-5,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.settings = dict(
            hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate, eps=eps, normalize=normalize
        )
        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(output_dim)
        self.fc3 = Dense(output_dim)  # input -> output residual adaptation
        self.ln = LayerNormalization(epsilon=eps) if normalize else None
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        out = tf.nn.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.dropout(out, training=training)
        out = out + self.fc3(x)
        return self.ln(out) if self.ln is not None else out

    def get_config(self):
        return dict(super().get_config(), **self.settings)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)[:-1].concatenate(self.settings["output_dim"])
