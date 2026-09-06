"""Behavioral forecast result and full-horizon transforms."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Optional, Tuple

import tensorflow as tf

from tfts.registry import register_transform


@dataclass(frozen=True)
class Trajectory:
    predictions: tf.Tensor
    samples: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None
    quantiles: Optional[Tuple[float, ...]] = None
    distribution: Optional[Any] = None

    def __post_init__(self):
        if self.predictions is None:
            raise ValueError("Trajectory.predictions is required")

    @property
    def mean(self):
        if self.samples is not None:
            return tf.reduce_mean(self.samples, axis=1)
        if self.distribution_params is not None and self.distribution is not None:
            return self.distribution.mean(self.distribution_params)
        return self.predictions

    def quantile(self, q: float):
        q = float(q)
        if not 0 <= q <= 1:
            raise ValueError("q must lie in [0, 1]")
        if self.quantile_values is not None and self.quantiles:
            if len(self.quantiles) == 1:
                return self.quantile_values[..., 0]
            if q in self.quantiles:
                return self.quantile_values[..., self.quantiles.index(q)]
            levels = tf.constant(self.quantiles, self.quantile_values.dtype)
            upper = tf.searchsorted(levels, tf.cast([q], levels.dtype), side="left")[0]
            upper = tf.clip_by_value(upper, 1, len(self.quantiles) - 1)
            lower = upper - 1
            q0, q1 = levels[lower], levels[upper]
            v0 = tf.gather(self.quantile_values, lower, axis=-1)
            v1 = tf.gather(self.quantile_values, upper, axis=-1)
            return v0 + (tf.cast(q, v0.dtype) - q0) / (q1 - q0) * (v1 - v0)
        if self.samples is not None:
            ordered = tf.sort(self.samples, axis=1)
            count = tf.shape(ordered)[1]
            index = tf.cast(tf.round(tf.cast(count - 1, tf.float32) * q), tf.int32)
            return ordered[:, index, ...]
        if self.distribution_params is not None and self.distribution is not None:
            quantile = getattr(self.distribution, "quantile", None)
            if quantile is not None:
                return quantile(self.distribution_params, q)
        return self.predictions

    def to_quantiles(self, qs):
        qs = tuple(float(q) for q in qs)
        values = tf.stack([self.quantile(q) for q in qs], axis=-1)
        return self.replace(quantile_values=values, quantiles=qs)

    def replace(self, **changes):
        return replace(self, **changes)

    def numpy(self):
        def convert(value):
            if tf.is_tensor(value):
                return value.numpy()
            if isinstance(value, Mapping):
                return {key: convert(item) for key, item in value.items()}
            return value

        values = {field.name: convert(getattr(self, field.name)) for field in fields(self)}
        return type(self)(**values)


class TrajectoryTransform:
    requires_fit = False

    def fit(self, trajectories, targets):
        return self

    def _check_fitted(self):
        if self.requires_fit and not getattr(self, "is_fitted", False):
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")


@register_transform("mean_samples")
class MeanSamples(TrajectoryTransform):
    def __call__(self, trajectory):
        if trajectory.samples is None:
            return trajectory
        return trajectory.replace(predictions=tf.reduce_mean(trajectory.samples, axis=1))


@register_transform("median_samples")
class MedianSamples(TrajectoryTransform):
    def __call__(self, trajectory):
        if trajectory.samples is None:
            return trajectory
        ordered = tf.sort(trajectory.samples, axis=1)
        count = tf.shape(ordered)[1]
        median = (ordered[:, (count - 1) // 2, ...] + ordered[:, count // 2, ...]) / 2.0
        return trajectory.replace(predictions=median)


@register_transform("inverse_scale")
class InverseScale(TrajectoryTransform):
    def __init__(self, scaler=None):
        self.scaler = scaler

    def __call__(self, trajectory):
        if self.scaler is None:
            return trajectory

        def inverse(value):
            return None if value is None else self.scaler.inverse_transform(value)

        return trajectory.replace(
            predictions=inverse(trajectory.predictions),
            samples=inverse(trajectory.samples),
            quantile_values=inverse(trajectory.quantile_values),
        )


@register_transform("repair_quantile_crossing")
class RepairQuantileCrossing(TrajectoryTransform):
    def __call__(self, trajectory):
        if trajectory.quantile_values is None:
            return trajectory
        return trajectory.replace(quantile_values=tf.sort(trajectory.quantile_values, axis=-1))


@register_transform("reconcile_hierarchy")
class ReconcileHierarchy(TrajectoryTransform):
    def __init__(self, S):
        self.S = tf.convert_to_tensor(S)

    def __call__(self, trajectory):
        predictions = tf.einsum("ij,bhj->bhi", self.S, trajectory.predictions)
        samples = None
        if trajectory.samples is not None:
            samples = tf.einsum("ij,bshj->bshi", self.S, trajectory.samples)
        return trajectory.replace(predictions=predictions, samples=samples)


@register_transform("conformal_calibrate")
class ConformalCalibrate(TrajectoryTransform):
    requires_fit = True

    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        self.is_fitted = False
        self.radius = None

    def fit(self, trajectories, targets):
        residual = tf.abs(tf.convert_to_tensor(targets) - trajectories.mean)
        ordered = tf.sort(tf.reshape(residual, [-1]))
        index = tf.cast(tf.round((1.0 - self.alpha) * tf.cast(tf.size(ordered) - 1, tf.float32)), tf.int32)
        self.radius = ordered[index]
        self.is_fitted = True
        return self

    def __call__(self, trajectory):
        self._check_fitted()
        values = tf.stack([trajectory.mean - self.radius, trajectory.mean + self.radius], axis=-1)
        return trajectory.replace(quantile_values=values, quantiles=(self.alpha / 2, 1 - self.alpha / 2))


@register_transform("clip")
class Clip(TrajectoryTransform):
    def __init__(self, minimum=None, maximum=None):
        if minimum is None and maximum is None:
            raise ValueError("minimum or maximum is required")
        self.minimum, self.maximum = minimum, maximum

    def __call__(self, trajectory):
        def clipped(value):
            if value is None:
                return None
            low = self.minimum if self.minimum is not None else tf.reduce_min(value)
            high = self.maximum if self.maximum is not None else tf.reduce_max(value)
            return tf.clip_by_value(value, tf.cast(low, value.dtype), tf.cast(high, value.dtype))

        return trajectory.replace(
            predictions=clipped(trajectory.predictions),
            samples=clipped(trajectory.samples),
            quantile_values=clipped(trajectory.quantile_values),
        )


__all__ = [
    "Clip",
    "ConformalCalibrate",
    "InverseScale",
    "MeanSamples",
    "MedianSamples",
    "ReconcileHierarchy",
    "RepairQuantileCrossing",
    "Trajectory",
    "TrajectoryTransform",
]
