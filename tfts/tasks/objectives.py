"""Task losses whose contract includes the complete time-series batch."""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

import tensorflow as tf

from tfts.batch import TimeSeriesBatch
from tfts.models.base import ModelOutput
from tfts.registry import register_objective


class LossResult(NamedTuple):
    """Reduction statistics; epoch loss is sum(total) / sum(weight)."""

    total: tf.Tensor
    weight: tf.Tensor

    @property
    def value(self):
        return tf.math.divide_no_nan(self.total, self.weight)


class Objective(Protocol):
    def reduce(self, batch: TimeSeriesBatch, output: ModelOutput) -> LossResult: ...  # noqa: E704


class ReducedObjective:
    """Keep scalar calls convenient while exposing exact reduction statistics."""

    requires_targets = True
    supports_sample_weight = True

    def __call__(self, batch, output):
        return self.reduce(batch, output).value


def _weighted_mean(values, mask=None):
    values = tf.cast(values, tf.float32)
    weight = tf.ones_like(values) if mask is None else tf.cast(mask, tf.float32)
    tf.debugging.assert_equal(tf.shape(weight), tf.shape(values), message="Loss weights must match loss elements")
    tf.debugging.assert_all_finite(weight, "Loss weights must be finite")
    tf.debugging.assert_non_negative(weight)
    values = tf.where(weight > 0, values, tf.zeros_like(values))
    tf.debugging.assert_all_finite(values, "Non-finite loss on observed targets")
    return LossResult(tf.reduce_sum(values * weight), tf.reduce_sum(weight))


def _safe_target(target, prediction, weight):
    target, prediction = tf.cast(target, tf.float32), tf.cast(prediction, tf.float32)
    tf.debugging.assert_equal(
        tf.shape(target), tf.shape(prediction), message="Targets and predictions must have identical shapes"
    )
    if weight is not None:
        # Mask operands before arithmetic, including its gradient computation.
        target = tf.where(weight > 0, target, tf.zeros_like(target))
        prediction = tf.where(weight > 0, prediction, tf.zeros_like(prediction))
    return target, prediction


def _forecast_target(batch, output):
    if batch.future_values is None:
        raise ValueError("forecast objective requires future_values")
    tf.debugging.assert_equal(
        tf.shape(batch.future_values),
        tf.shape(output.predictions),
        message="Targets and predictions must have identical shapes",
    )
    return tf.cast(batch.future_values, tf.float32), tf.cast(output.predictions, tf.float32)


def point_weights(batch, target, mask_field="future_observed_mask"):
    """Combine observation masks and sample weights without implicit broadcasting."""
    weight = batch.sample_weight
    if weight is not None:
        weight = tf.cast(weight, target.dtype)
        rank = weight.shape.rank
        if rank is None or rank < 1 or rank > target.shape.rank:
            raise ValueError("sample_weight must have shape (B,), (B,T), or (B,T,D)")
        tf.debugging.assert_equal(
            tf.shape(weight), tf.shape(target)[:rank], message="sample_weight shape does not match targets"
        )
        weight = tf.reshape(weight, tf.concat([tf.shape(weight), tf.ones([target.shape.rank - rank], tf.int32)], 0))
        weight = tf.broadcast_to(weight, tf.shape(target))
        tf.debugging.assert_all_finite(weight, "sample_weight must be finite")
        tf.debugging.assert_non_negative(weight)
    observed = getattr(batch, mask_field) if mask_field is not None else None
    if observed is not None:
        mask = tf.cast(observed, target.dtype)
        tf.debugging.assert_equal(tf.shape(mask), tf.shape(target))
        weight = mask if weight is None else weight * mask
    return weight


@register_objective("sequence_mse")
class SequenceMeanSquaredError(ReducedObjective):
    requires_targets = True
    supports_sample_weight = True

    def reduce(self, batch, output):
        target = tf.cast(batch.labels, tf.float32)
        tf.debugging.assert_equal(
            tf.shape(target), tf.shape(output.predictions), message="Targets and predictions must have identical shapes"
        )
        weight = point_weights(batch, target)
        target, prediction = _safe_target(target, output.predictions, weight)
        return _weighted_mean(tf.square(target - prediction), weight)


@register_objective("mse")
class MeanSquaredError(ReducedObjective):
    requires_targets = True
    supports_sample_weight = True

    def reduce(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        weight = point_weights(batch, target)
        target, prediction = _safe_target(target, prediction, weight)
        return _weighted_mean(tf.square(target - prediction), weight)


@register_objective("mae")
class MeanAbsoluteError(ReducedObjective):
    requires_targets = True
    supports_sample_weight = True

    def reduce(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        weight = point_weights(batch, target)
        target, prediction = _safe_target(target, prediction, weight)
        return _weighted_mean(tf.abs(target - prediction), weight)


@register_objective("smape")
class SymmetricMeanAbsolutePercentageError(ReducedObjective):
    requires_targets = True

    def __init__(self, epsilon=1e-7):
        self.epsilon = float(epsilon)

    def reduce(self, batch, output):
        target, prediction = _forecast_target(batch, output)
        weight = point_weights(batch, target)
        target, prediction = _safe_target(target, prediction, weight)
        denominator = tf.maximum(tf.abs(target) + tf.abs(prediction), self.epsilon)
        return _weighted_mean(2.0 * tf.abs(target - prediction) / denominator, weight)


@register_objective("quantile")
class QuantileObjective(ReducedObjective):
    requires_targets = True

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        self.quantiles = tuple(float(q) for q in quantiles)

    def reduce(self, batch, output):
        if batch.future_values is None or output.quantile_values is None:
            raise ValueError("quantile objective requires future_values and quantile_values")
        values = tf.cast(output.quantile_values, tf.float32)
        target = tf.cast(batch.future_values, tf.float32)
        tf.debugging.assert_equal(tf.shape(values)[:-1], tf.shape(target))
        tf.debugging.assert_equal(tf.shape(values)[-1], len(self.quantiles))
        weight = point_weights(batch, target)
        target = target[..., None]
        if weight is not None:
            active = weight[..., None] > 0
            target = tf.where(active, target, tf.zeros_like(target))
            values = tf.where(active, values, tf.zeros_like(values))
        error = target - values
        q = tf.constant(self.quantiles, dtype=tf.float32)
        losses = tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error), axis=-1)
        return _weighted_mean(losses, weight)


@register_objective("nll")
class NegativeLogLikelihood(ReducedObjective):
    requires_targets = True

    def __init__(self, distribution=None):
        self.distribution = distribution

    def reduce(self, batch, output):
        if self.distribution is None or output.distribution_params is None:
            raise ValueError("nll objective requires a distribution and distribution_params")
        if batch.future_values is None:
            raise ValueError("nll objective requires future_values")
        target = tf.cast(batch.future_values, tf.float32)
        weight = point_weights(batch, target)
        parameters = {key: tf.cast(value, tf.float32) for key, value in output.distribution_params.items()}
        tf.debugging.assert_equal(tf.shape(target), tf.shape(output.predictions))
        if weight is not None:
            target = tf.where(weight > 0, target, tf.zeros_like(target))
        losses = self.distribution.loss(target, parameters, reduction="none")
        return _weighted_mean(losses, weight)


@register_objective("crps")
class GaussianCRPS(ReducedObjective):
    """Closed-form CRPS for the normal distribution parameter contract."""

    requires_targets = True

    def reduce(self, batch, output):
        if batch.future_values is None or output.distribution_params is None:
            raise ValueError("crps objective requires future_values and distribution_params")
        loc = tf.cast(output.distribution_params["loc"], tf.float32)
        scale = tf.cast(output.distribution_params["scale"], tf.float32)
        target = tf.cast(batch.future_values, tf.float32)
        weight = point_weights(batch, target)
        target, loc = _safe_target(target, loc, weight)
        z = (target - loc) / scale
        normal_pdf = tf.exp(-0.5 * tf.square(z)) / tf.sqrt(tf.cast(2.0 * math.pi, z.dtype))
        normal_cdf = 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(tf.cast(2.0, z.dtype))))
        score = scale * (z * (2.0 * normal_cdf - 1.0) + 2.0 * normal_pdf - 1.0 / tf.sqrt(tf.constant(math.pi)))
        return _weighted_mean(score, weight)


@register_objective("masked_mse")
class MaskedReconstructionMSE(ReducedObjective):
    requires_targets = True

    def __init__(self, selection="missing", target_field="labels", prediction_field="reconstructed_values"):
        if selection not in {"missing", "observed"}:
            raise ValueError("selection must be 'missing' or 'observed'")
        self.selection = selection
        self.target_field = target_field
        self.prediction_field = prediction_field

    def reduce(self, batch, output):
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
        weight = point_weights(batch, tf.cast(target, tf.float32), mask_field=None)
        mask = tf.cast(mask, tf.float32)
        weight = mask if weight is None else weight * mask
        target, prediction = _safe_target(target, prediction, weight)
        return _weighted_mean(tf.square(target - prediction), weight)


@register_objective("ce")
class CrossEntropy(ReducedObjective):
    requires_targets = True

    def reduce(self, batch, output):
        if batch.labels is None or output.logits is None:
            raise ValueError("ce objective requires labels and logits")
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch.labels, output.logits, from_logits=True)
        return _weighted_mean(losses, point_weights(batch, losses, mask_field=None))


@register_objective("masked_ce")
class MaskedCrossEntropy(ReducedObjective):
    requires_targets = True

    def reduce(self, batch, output):
        if batch.labels is None or output.logits is None:
            raise ValueError("masked_ce objective requires labels and logits")
        losses = tf.keras.losses.sparse_categorical_crossentropy(batch.labels, output.logits, from_logits=True)
        mask = batch.future_observed_mask
        if mask is None:
            mask = batch.past_observed_mask
        if mask is not None and mask.shape.rank == losses.shape.rank + 1:
            mask = tf.reduce_any(tf.cast(mask, tf.bool), axis=-1)
        weight = point_weights(batch, losses, mask_field=None)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            weight = mask if weight is None else weight * mask
        return _weighted_mean(losses, weight)


__all__ = [
    "CrossEntropy",
    "GaussianCRPS",
    "MaskedCrossEntropy",
    "MaskedReconstructionMSE",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "NegativeLogLikelihood",
    "Objective",
    "LossResult",
    "ReducedObjective",
    "QuantileObjective",
    "SymmetricMeanAbsolutePercentageError",
]
