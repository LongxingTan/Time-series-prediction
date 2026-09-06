"""Task losses whose contract includes the complete time-series batch."""

from __future__ import annotations

import math
from typing import Optional, Protocol

import tensorflow as tf

from tfts.contracts import ModelOutput, TimeSeriesBatch
from tfts.registry import register_objective


class Objective(Protocol):
    requires_targets: bool

    def __call__(self, batch: TimeSeriesBatch, output: ModelOutput) -> tf.Tensor: ...  # noqa: E704


def _weighted_mean(values, mask=None):
    values = tf.convert_to_tensor(values)
    if mask is None:
        return tf.reduce_mean(values)
    weight = tf.cast(mask, values.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(values * weight), tf.reduce_sum(weight))


def _forecast_target(batch, output):
    if batch.future_values is None:
        raise ValueError("forecast objective requires future_values")
    return tf.cast(batch.future_values, output.predictions.dtype), output.predictions


@register_objective("mse")
class MeanSquaredError:
    requires_targets = True

    def __call__(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        return _weighted_mean(tf.square(target - prediction), batch.future_observed_mask)


@register_objective("mae")
class MeanAbsoluteError:
    requires_targets = True

    def __call__(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        return _weighted_mean(tf.abs(target - prediction), batch.future_observed_mask)


@register_objective("smape")
class SymmetricMeanAbsolutePercentageError:
    requires_targets = True

    def __init__(self, epsilon=1e-7):
        self.epsilon = float(epsilon)

    def __call__(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        denominator = tf.maximum(tf.abs(target) + tf.abs(prediction), self.epsilon)
        return _weighted_mean(2.0 * tf.abs(target - prediction) / denominator, batch.future_observed_mask)


@register_objective("quantile")
class QuantileObjective:
    requires_targets = True

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        self.quantiles = tuple(float(q) for q in quantiles)

    def __call__(self, batch, output):
        if batch.future_values is None or output.quantile_values is None:
            raise ValueError("quantile objective requires future_values and quantile_values")
        values = output.quantile_values
        target = tf.expand_dims(tf.cast(batch.future_values, values.dtype), -1)
        error = target - values
        q = tf.cast(self.quantiles, values.dtype)
        losses = tf.maximum(q * error, (q - 1.0) * error)
        mask = None
        if batch.future_observed_mask is not None:
            mask = tf.expand_dims(batch.future_observed_mask, -1)
        return _weighted_mean(losses, mask)


@register_objective("nll")
class NegativeLogLikelihood:
    requires_targets = True

    def __init__(self, distribution=None):
        self.distribution = distribution

    def __call__(self, batch, output):
        if self.distribution is None or output.distribution_params is None:
            raise ValueError("nll objective requires a distribution and distribution_params")
        if batch.future_values is None:
            raise ValueError("nll objective requires future_values")
        losses = self.distribution.loss(
            tf.cast(batch.future_values, output.predictions.dtype), output.distribution_params, reduction="none"
        )
        return _weighted_mean(losses, batch.future_observed_mask)


@register_objective("crps")
class GaussianCRPS:
    """Closed-form CRPS for the normal distribution parameter contract."""

    requires_targets = True

    def __call__(self, batch, output):
        if batch.future_values is None or output.distribution_params is None:
            raise ValueError("crps objective requires future_values and distribution_params")
        loc = output.distribution_params["loc"]
        scale = output.distribution_params["scale"]
        z = (tf.cast(batch.future_values, loc.dtype) - loc) / scale
        normal_pdf = tf.exp(-0.5 * tf.square(z)) / tf.sqrt(tf.cast(2.0 * math.pi, z.dtype))
        normal_cdf = 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(tf.cast(2.0, z.dtype))))
        score = scale * (z * (2.0 * normal_cdf - 1.0) + 2.0 * normal_pdf - 1.0 / tf.sqrt(tf.constant(math.pi)))
        return _weighted_mean(score, batch.future_observed_mask)


@register_objective("masked_mse")
class MaskedReconstructionMSE:
    requires_targets = True

    def __init__(self, selection="missing", target_field="labels", prediction_field="reconstructed_values"):
        if selection not in {"missing", "observed"}:
            raise ValueError("selection must be 'missing' or 'observed'")
        self.selection = selection
        self.target_field = target_field
        self.prediction_field = prediction_field

    def __call__(self, batch, output):
        target = getattr(batch, self.target_field)
        prediction = getattr(output, self.prediction_field)
        if target is None or prediction is None:
            raise ValueError("masked_mse objective is missing its target or prediction")
        observed = batch.past_observed_mask
        if observed is None:
            if self.selection == "missing":
                raise ValueError("masked_mse objective requires past_observed_mask")
            observed = tf.ones_like(prediction, dtype=tf.bool)
        mask = observed if self.selection == "observed" else tf.logical_not(tf.cast(observed, tf.bool))
        return _weighted_mean(tf.square(tf.cast(target, prediction.dtype) - prediction), mask)


@register_objective("ce")
class CrossEntropy:
    requires_targets = True

    def __call__(self, batch, output):
        if batch.labels is None or output.logits is None:
            raise ValueError("ce objective requires labels and logits")
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch.labels, output.logits, from_logits=True)
        return tf.reduce_mean(losses)


@register_objective("masked_ce")
class MaskedCrossEntropy:
    requires_targets = True

    def __call__(self, batch, output):
        if batch.labels is None or output.logits is None:
            raise ValueError("masked_ce objective requires labels and logits")
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch.labels, output.logits, from_logits=True)
        mask = batch.future_observed_mask
        if mask is None:
            mask = batch.past_observed_mask
        if mask is not None and mask.shape.rank == losses.shape.rank + 1:
            mask = tf.reduce_any(tf.cast(mask, tf.bool), axis=-1)
        return _weighted_mean(losses, mask)


__all__ = [
    "CrossEntropy",
    "GaussianCRPS",
    "MaskedCrossEntropy",
    "MaskedReconstructionMSE",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "NegativeLogLikelihood",
    "Objective",
    "QuantileObjective",
    "SymmetricMeanAbsolutePercentageError",
]
