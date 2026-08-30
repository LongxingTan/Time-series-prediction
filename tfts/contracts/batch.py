"""Canonical model input for every TFTS task."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple

import tensorflow as tf

from .structure import ARRANGEMENT_BY_RANK, EXPECTED_RANK, SpatialArrangement, SpatialStructure


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
    structure: Optional[SpatialStructure] = None

    def __post_init__(self) -> None:
        if self.past_values is None:
            raise ValueError("past_values is required")
        self.past_values = tf.convert_to_tensor(self.past_values)
        rank = self.past_values.shape.rank
        if rank not in ARRANGEMENT_BY_RANK:
            raise ValueError("past_values must have rank 3 (sequence), 4 (set), or 5 (grid), " f"got rank {rank}")
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
        if self.structure is not None:
            self.structure.validate(self.past_values)

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
            inputs = dict(inputs)
            structure_values = {key: value for key, value in inputs.items() if key.startswith("structure.")}
            for key in structure_values:
                inputs.pop(key)
            if structure_values:
                inputs["structure"] = SpatialStructure.from_tensor_dict(structure_values)
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

    def as_tensor_dict(self, include_structure: bool = True) -> Dict[str, tf.Tensor]:
        """Return tensor leaves only for Keras and ``tf.data`` boundaries."""
        values = {
            field.name: getattr(self, field.name) for field in fields(self) if tf.is_tensor(getattr(self, field.name))
        }
        if include_structure and self.structure is not None:
            values.update(self.structure.to_tensor_dict())
        return values

    @property
    def arrangement(self) -> SpatialArrangement:
        return ARRANGEMENT_BY_RANK[self.past_values.shape.rank]

    @property
    def topology_inputs(self):
        return self.structure.topology_inputs if self.structure is not None else frozenset()

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        dimensions = self.past_values.shape[2:-1]
        if any(dimension is None for dimension in dimensions):
            if self.structure is None:
                raise ValueError("spatial dimensions must be statically known when no structure is provided")
            return self.structure.spatial_shape
        return tuple(int(dimension) for dimension in dimensions)

    @property
    def spatial_axes(self) -> Tuple[int, ...]:
        return tuple(range(2, 2 + len(self.spatial_shape)))

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
        spatial_rank = EXPECTED_RANK[self.arrangement]
        temporal_fields = (
            ("past_time_features", self.past_time_features, self.context_length),
            ("past_categorical_features", self.past_categorical_features, self.context_length),
        )
        for name, value, expected_length in temporal_fields:
            if value is not None:
                if value.shape.rank not in {3, spatial_rank}:
                    raise ValueError(f"{name} must be shared rank-3 or match the batch layout")
                tf.debugging.assert_equal(tf.shape(value)[0], self.batch_size, message=f"{name} batch size mismatch")
                tf.debugging.assert_equal(tf.shape(value)[1], expected_length, message=f"{name} time length mismatch")

        future_lengths = []
        for name, value in (
            ("future_time_features", self.future_time_features),
            ("future_categorical_features", self.future_categorical_features),
            ("future_values", self.future_values),
        ):
            if value is not None:
                if value.shape.rank not in {3, spatial_rank}:
                    raise ValueError(f"{name} must be shared rank-3 or match the batch layout")
                tf.debugging.assert_equal(tf.shape(value)[0], self.batch_size, message=f"{name} batch size mismatch")
                future_lengths.append((name, tf.shape(value)[1]))
        for name, length in future_lengths[1:]:
            tf.debugging.assert_equal(length, future_lengths[0][1], message=f"{name} horizon mismatch")

        for name, value, reference in (
            ("past_observed_mask", self.past_observed_mask, self.past_values),
            ("future_observed_mask", self.future_observed_mask, self.future_values),
        ):
            if value is not None and reference is not None:
                tf.debugging.assert_equal(
                    tf.shape(value), tf.shape(reference), message=f"{name} must have the same shape as its values"
                )

        if task == "imputation":
            if self.past_observed_mask is None:
                raise ValueError("imputation requires past_observed_mask")
            tf.debugging.assert_equal(
                tf.shape(self.past_observed_mask),
                tf.shape(self.past_values),
                message="past_observed_mask must have the same shape as past_values",
            )
        elif task == "classification" and self.labels is not None:
            if self.labels.shape.rank not in (1, 2):
                raise ValueError("classification labels must have rank 1 or 2")
