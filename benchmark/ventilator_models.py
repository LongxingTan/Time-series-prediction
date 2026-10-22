"""TensorFlow translation of theoviel's simple LSTM baseline and cdeotte's transformer.

The reference transformer follows cdeotte's "tensorflow-transformer" notebook:
an input projection, a stack of pre-norm MultiHeadAttention transformer blocks
with a residual weighting, and a per-timestep multi-channel ``[pressure,
pressure_diff, pressure_integral]`` head. The masked loss and the evaluation in
``benchmark.ventilator_bert`` match the notebook's transient sample weighting.

The remaining names are general TFTS architectures running the aligned
per-step protocol from ``benchmark/datasets/ventilator.py``: five raw inputs and
one pressure per step. ``ALIGNED_TFTS_MODELS`` lists the architectures whose
backbone already returns ``(batch, 80, 1)``. Architectures that forecast every
input channel (``itransformer``, ``diffusion``, ``timemixer``, ``timesnet``) or
that cannot accept a plain ``(batch, time, features)`` tensor (``stgcn``,
``tft``, ``deep_ar``, ``nbeats``, ``timexer``) are rejected here with the
reason, rather than silently scoring a mis-shaped tensor.
"""

from __future__ import annotations

from typing import Mapping, Union

import tensorflow as tf

from tfts.models.bert import Bert, BertConfig
from tfts.models.rnn import RNN, RNNConfig

ModelParameters = Mapping[str, Union[int, float, bool, str]]

ALIGNED_TFTS_MODELS = (
    "autoformer",
    "bert",
    "dlinear",
    "gpt",
    "informer",
    "patch_tst",
    "rnn",
    "rwkv",
    "tcn",
    "tide",
    "timesfm",
)

UNSUPPORTED_TFTS_MODELS = {
    "itransformer": "forecasts every input channel and exposes no sequence port for a point head",
    "diffusion": "forecasts every input channel and exposes no sequence port for a point head",
    "timemixer": "forecasts every input channel; use a point head through the task layer",
    "timesnet": "forecasts every input channel; use a point head through the task layer",
    "timexer": "decoder length is patch_num + 1, so it cannot emit an aligned 80-step sequence",
    "deep_ar": "tiles a series embedding onto a single input channel, so multivariate input breaks the encoder width",
    "nbeats": "squeezes the channel axis and accepts only univariate input",
    "stgcn": "requires the 'set' input arrangement instead of a past-values tensor",
    "tft": "requires explicit encoder/decoder real and categorical dimensions",
    "seq2seq": "decodes autoregressively, so every prediction depends on later input steps no causal meaning here",
}


class VentilatorRunError(ValueError):
    """A benchmark configuration cannot produce a valid comparison."""


def build_aligned_tfts_model(name: str, overrides: ModelParameters | None = None) -> tf.keras.Model:
    """Build a TFTS backbone that returns one value per input step."""
    from tfts.models.auto_config import AutoConfig
    from tfts.models.auto_model import AutoBackbone

    if name in UNSUPPORTED_TFTS_MODELS:
        raise VentilatorRunError(f"{name} cannot use the aligned protocol: {UNSUPPORTED_TFTS_MODELS[name]}")
    if name not in ALIGNED_TFTS_MODELS:
        raise VentilatorRunError(f"'{name}' is not an aligned TFTS architecture")
    config = AutoConfig.for_model(name)
    config.model_type = name
    unknown = set(overrides or {}) - set(config.to_dict())
    if unknown:
        raise VentilatorRunError(f"Unsupported {name} overrides: {sorted(unknown)}")
    config.update(dict(overrides or {}))
    config.target_dim = 1
    model = AutoBackbone.from_config(config, output_chunk_length=80)
    if model.config.target_dim != 1:
        raise VentilatorRunError(f"{name} must predict a single channel")
    return model


def build_ventilator_model(name: str, num_features: int, overrides: ModelParameters | None = None) -> tf.keras.Model:
    """Construct the same architecture for training, resume, and verification."""
    parameters = dict(overrides or {})
    if name == "lstm_baseline":
        return build_lstm_baseline(**parameters)
    if name == "rnn":
        return RNN(80, RNNConfig(**parameters))
    if name == "reference":
        if parameters.get("num_targets", 3) != 3:
            raise VentilatorRunError("Reference targets must have three channels")
        return ReferenceTransformer(num_features=num_features, **parameters)
    if name == "bert":
        defaults = {
            "hidden_size": 64,
            "num_layers": 4,
            "num_attention_heads": 8,
            "ffn_intermediate_size": 128,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "regression_head": True,
            "regression_head_units": 128,
            "output_size": 3,
            "positional_type": "positional encoding",
        }
        unknown = parameters.keys() - defaults.keys()
        if unknown:
            raise VentilatorRunError(f"Unsupported BERT overrides: {sorted(unknown)}")
        defaults.update(parameters)
        if not defaults["regression_head"] or defaults["output_size"] != 3:
            raise VentilatorRunError("BERT requires aligned regression with three output channels")
        return Bert(predict_sequence_length=80, config=BertConfig(**defaults))
    if name in UNSUPPORTED_TFTS_MODELS:
        raise VentilatorRunError(f"{name} cannot use the aligned protocol: {UNSUPPORTED_TFTS_MODELS[name]}")
    if name in ALIGNED_TFTS_MODELS:
        return build_aligned_tfts_model(name, parameters)
    raise VentilatorRunError(f"Unsupported ventilator model: {name}")


def build_lstm_baseline(dense_dim: int = 512, lstm_dim: int = 512, logit_dim: int = 512) -> tf.keras.Model:
    """Preserve the notebook's MLP, bidirectional LSTM and per-step head."""
    inputs = tf.keras.Input(shape=(80, 5))
    features = tf.keras.layers.Dense(dense_dim // 2, activation="relu")(inputs)
    features = tf.keras.layers.Dense(dense_dim, activation="relu")(features)
    features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True))(features)
    features = tf.keras.layers.Dense(logit_dim, activation="relu")(features)
    outputs = tf.keras.layers.Dense(1)(features)
    return tf.keras.Model(inputs, outputs, name="theoviel_lstm")


def ventilator_loss(packed_target: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """Mean per-breath inspiration MAE, matching the reference training loss.

    Targets pack pressure and the inspiration mask in the last dimension.
    Evaluation must separately use the global inspiration-weighted MAE.
    """
    pressure = packed_target[..., :1]
    mask = packed_target[..., 1:2]
    return tf.math.divide_no_nan(
        tf.reduce_sum(tf.abs(pressure - prediction) * mask, axis=(1, 2)),
        tf.reduce_sum(mask, axis=(1, 2)),
    )


class ReferenceTransformerBlock(tf.keras.layers.Layer):
    """cdeotte's transformer encoder block: pre-norm MHA plus an MLP expansion."""

    def __init__(self, embed_dim: int, feat_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="gelu"), tf.keras.layers.Dense(feat_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs: tf.Tensor, training=None, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "feat_dim": self.feat_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(name="ReferenceTransformer")
class ReferenceTransformer(tf.keras.Model):
    """cdeotte's ventilator transformer: 12 blocks, residual weighting, 3-channel head."""

    def __init__(
        self,
        num_features: int,
        feat_dim_extra: int = 32,
        embed_dim: int = 64,
        num_heads: int = 8,
        ff_dim: int = 128,
        dropout_rate: float = 0.0,
        num_blocks: int = 12,
        head_units: int = 128,
        num_targets: int = 3,
        **kwargs,
    ):
        kwargs.setdefault("name", "reference_transformer")
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feat_dim = num_features + feat_dim_extra
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.head_units = head_units
        self.num_targets = num_targets
        self.embedding = tf.keras.layers.Dense(self.feat_dim)
        self.embed_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.blocks = [
            ReferenceTransformerBlock(embed_dim, self.feat_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        self.head_dense = tf.keras.layers.Dense(head_units, activation="selu")
        self.head_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.head_out = tf.keras.layers.Dense(num_targets, activation="linear")

    def call(self, inputs: tf.Tensor, training=None, **kwargs) -> tf.Tensor:
        x = self.embedding(inputs)
        x = self.embed_ln(x)
        for block in self.blocks:
            x_old = x
            x = block(x, training=training)
            x = 0.7 * x + 0.3 * x_old
        x = self.head_dense(x)
        x = self.head_dropout(x, training=training)
        return self.head_out(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "feat_dim_extra": self.feat_dim - self.num_features,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "num_blocks": self.num_blocks,
                "head_units": self.head_units,
                "num_targets": self.num_targets,
            }
        )
        return config


def masked_multi_mae(packed_target: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """Global inspiration-masked MAE across the multi-channel target stack.

    ``packed_target`` concatenates ``[target, mask]`` along the last axis where
    ``mask`` has shape ``(batch, 1)``. This mirrors the notebook's transient
    sample weighting: expiration rows contribute nothing to training.
    """
    targets = packed_target[..., :-1]
    weight = packed_target[..., -1:]
    weight = tf.broadcast_to(weight, tf.shape(targets))
    numerator = tf.reduce_sum(tf.abs(targets - prediction) * weight)
    denominator = tf.maximum(tf.reduce_sum(weight), 1.0)
    return numerator / denominator
