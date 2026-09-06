"""Shared input boundary for native continuous autoregressive backbones."""

import tensorflow as tf

from tfts.contracts import BackboneCapabilities, ForecastMode, ModelInputSpec, TimeSeriesBatch
from tfts.generation.decoding import decode

from .base import BaseModel

AUTOREGRESSIVE_CAPABILITIES = BackboneCapabilities(
    forecast_modes=frozenset({ForecastMode.AUTOREGRESSIVE}),
    supports_future_covariates=True,
    input_spec=ModelInputSpec(accepted_roles=frozenset({"observed_past", "known_future"})),
)


def decoder_features(batch, horizon):
    """Future covariates, or a relative forecast-position feature."""
    features = batch.future_time_features
    if features is None:
        features = tf.tile(tf.range(horizon)[None, :, None], [batch.batch_size, 1, 1])
    tf.debugging.assert_greater_equal(tf.shape(features)[1], horizon, message="future covariates are too short")
    return tf.cast(features[:, :horizon, :], batch.past_values.dtype)


def encoder_features(batch):
    values = batch.past_values
    if batch.past_time_features is not None:
        values = tf.concat([values, tf.cast(batch.past_time_features, values.dtype)], axis=-1)
    return values


class AutoregressiveModel(BaseModel):
    """Legacy tensor calls and canonical generation share the same decoder."""

    def decoder_seed(self, batch):
        """Seed feedback with target channels, retaining all inputs for encoding.

        Legacy arrays can contain additional observed channels after the targets.
        The seed must have the same width as subsequent decoder predictions.
        """
        target_dim = self.config.target_dim
        tf.debugging.assert_greater_equal(
            tf.shape(batch.past_values)[-1], target_dim, message="history has fewer channels than target_dim"
        )
        return batch.past_values[:, -1:, :target_dim]

    def call(
        self, inputs, teacher=None, training=None, teacher_probability=None, output_hidden_states=None, return_dict=None
    ):
        if isinstance(inputs, dict) and "past_values" in inputs:
            batch = TimeSeriesBatch.from_inputs(inputs)
        else:
            x, encoder, future = self._prepare_3d_inputs(inputs, ignore_decoder_inputs=False)
            target_dim = self.config.target_dim
            batch = TimeSeriesBatch(
                past_values=x[..., :target_dim],
                past_time_features=encoder[..., target_dim:],
                future_time_features=future,
                future_values=teacher,
            )
        if teacher_probability is None:
            teacher_probability = 1.0 - self.config.scheduled_sampling if teacher is not None else 0.0
        output = decode(
            self, batch, self.predict_sequence_length, training=training, teacher_probability=teacher_probability
        )
        return output.predictions
