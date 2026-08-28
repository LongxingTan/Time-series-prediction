"""Canonical model input for every TFTS task."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

import tensorflow as tf


@dataclass
class TimeSeriesBatch:
    """Named time-series tensors shared by forecasting and representation tasks.

    Masks use one convention throughout TFTS: ``True``/``1`` means observed or
    valid. Missing numeric values must be filled before model execution.
    """

    past_values: tf.Tensor
    future_values: Optional[tf.Tensor] = None
    past_time_features: Optional[tf.Tensor] = None
    future_time_features: Optional[tf.Tensor] = None
    past_categorical_features: Optional[tf.Tensor] = None
    future_categorical_features: Optional[tf.Tensor] = None
    static_real_features: Optional[tf.Tensor] = None
    static_categorical_features: Optional[tf.Tensor] = None
    past_observed_mask: Optional[tf.Tensor] = None
    future_observed_mask: Optional[tf.Tensor] = None
    padding_mask: Optional[tf.Tensor] = None
    labels: Optional[tf.Tensor] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.past_values is None:
            raise ValueError("past_values is required")
        self.past_values = tf.convert_to_tensor(self.past_values)
        if self.past_values.shape.rank != 3:
            raise ValueError("past_values must have shape [batch, time, target]")
        for name in (
            "future_values",
            "past_time_features",
            "future_time_features",
            "past_categorical_features",
            "future_categorical_features",
            "static_real_features",
            "static_categorical_features",
            "past_observed_mask",
            "future_observed_mask",
            "padding_mask",
            "labels",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, tf.convert_to_tensor(value))

    @classmethod
    def from_inputs(cls, inputs: Any) -> "TimeSeriesBatch":
        """Normalize a canonical mapping, tensor, or existing batch.

        Plain tensors intentionally mean ``past_values``. Mappings must use
        :class:`TimeSeriesBatch` field names. Positional multi-tensor inputs are
        rejected because they cannot distinguish covariates, future targets,
        masks, or dataset-level ``(inputs, labels)`` pairs.
        """
        if isinstance(inputs, cls):
            return inputs
        if isinstance(inputs, Mapping):
            known = {field.name for field in fields(cls)}
            unknown = set(inputs) - known
            if unknown:
                raise ValueError("Unknown TimeSeriesBatch fields: %s" % sorted(unknown))
            return cls(**dict(inputs))
        if isinstance(inputs, (tuple, list)):
            raise ValueError("Positional time-series inputs are ambiguous; use canonical " "TimeSeriesBatch fields")
        return cls(past_values=inputs)

    def as_dict(self, include_none: bool = False) -> Dict[str, Any]:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        if include_none:
            return values
        return {name: value for name, value in values.items() if value is not None}

    @property
    def batch_size(self):
        return tf.shape(self.past_values)[0]

    @property
    def context_length(self):
        return tf.shape(self.past_values)[1]

    @property
    def target_dim(self):
        return tf.shape(self.past_values)[-1]

    def validate_for(self, task: str) -> None:
        temporal_fields = (
            ("past_time_features", self.past_time_features, self.context_length),
            ("past_categorical_features", self.past_categorical_features, self.context_length),
        )
        for name, value, expected_length in temporal_fields:
            if value is not None:
                if value.shape.rank != 3:
                    raise ValueError(f"{name} must have shape [batch, time, feature]")
                tf.debugging.assert_equal(tf.shape(value)[0], self.batch_size, message=f"{name} batch size mismatch")
                tf.debugging.assert_equal(tf.shape(value)[1], expected_length, message=f"{name} time length mismatch")

        future_lengths = []
        for name, value in (
            ("future_time_features", self.future_time_features),
            ("future_categorical_features", self.future_categorical_features),
            ("future_values", self.future_values),
        ):
            if value is not None:
                if value.shape.rank != 3:
                    raise ValueError(f"{name} must have shape [batch, horizon, feature]")
                tf.debugging.assert_equal(tf.shape(value)[0], self.batch_size, message=f"{name} batch size mismatch")
                future_lengths.append((name, tf.shape(value)[1]))
        for name, length in future_lengths[1:]:
            tf.debugging.assert_equal(length, future_lengths[0][1], message=f"{name} horizon mismatch")

        if task == "imputation":
            if self.past_observed_mask is None:
                raise ValueError("imputation requires past_observed_mask")
            if self.past_observed_mask.shape != self.past_values.shape:
                raise ValueError("past_observed_mask must have the same shape as past_values")
        elif task == "classification" and self.labels is not None:
            if self.labels.shape.rank not in (1, 2):
                raise ValueError("classification labels must have rank 1 or 2")
