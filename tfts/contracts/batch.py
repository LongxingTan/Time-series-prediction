"""Canonical model input for every TFTS task."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import tensorflow as tf

from .structure import ARRANGEMENT_BY_RANK, EXPECTED_RANK, SpatialArrangement, SpatialStructure


@dataclass(frozen=True)
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
        object.__setattr__(self, "past_values", tf.convert_to_tensor(self.past_values))
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
                object.__setattr__(self, name, tf.convert_to_tensor(value))
        if self.structure is not None:
            self.structure.validate(self.past_values)
        names = (self.metadata or {}).get("feature_names", {})
        for past_key, future_key, past, future in (
            ("past_real", "future_real", self.past_time_features, self.future_time_features),
            (
                "past_categorical",
                "future_categorical",
                self.past_categorical_features,
                self.future_categorical_features,
            ),
        ):
            for key, value in ((past_key, past), (future_key, future)):
                if key in names and value is not None:
                    if value.shape[-1] is not None and len(names[key]) != value.shape[-1]:
                        raise ValueError(f"{key} metadata does not match its tensor width")

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

        if self.padding_mask is not None:
            if self.padding_mask.shape.rank != 2:
                raise ValueError("padding_mask must have shape (batch, time)")
            tf.debugging.assert_equal(
                tf.shape(self.padding_mask),
                tf.shape(self.past_values)[:2],
                message="padding_mask must match the past batch and time dimensions",
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

    def advance(self, value: tf.Tensor, *, offset: tf.Tensor) -> "TimeSeriesBatch":
        """Roll the context forward and consume matching known-future features."""

        value = tf.convert_to_tensor(value)
        tf.debugging.assert_rank_at_least(value, 3)
        tf.debugging.assert_equal(tf.shape(value)[0], self.batch_size, message="feedback batch size mismatch")
        tf.debugging.assert_equal(tf.shape(value)[-1], self.target_dim, message="feedback target dimension mismatch")
        tf.debugging.assert_non_negative(offset, message="offset must be non-negative")
        width = tf.shape(value)[1]
        tf.debugging.assert_positive(width, message="feedback chunk must not be empty")

        def roll(current, appended):
            joined = tf.concat([current, tf.cast(appended, current.dtype)], axis=1)
            return joined[:, -tf.shape(current)[1] :, ...]

        names = (self.metadata or {}).get("feature_names", {})

        def advance_features(current, future, past_key, future_key, field_name):
            if current is None:
                return None
            if future is None:
                appended = tf.repeat(current[:, -1:, ...], width, axis=1)
            else:
                tf.debugging.assert_greater_equal(
                    tf.shape(future)[1], width, message=f"{field_name} do not cover feedback chunk"
                )
                appended = future[:, :width, ...]
                past_names = tuple(names.get(past_key, ()))
                future_names = tuple(names.get(future_key, ()))
                if past_names or future_names:
                    unknown = set(future_names) - set(past_names)
                    if unknown:
                        raise ValueError(f"future {field_name} are absent from the past layout: {sorted(unknown)}")
                    positions = {name: index for index, name in enumerate(future_names)}
                    columns = []
                    for index, name in enumerate(past_names):
                        if name in positions:
                            position = positions[name]
                            columns.append(appended[..., position : position + 1])
                        else:
                            columns.append(tf.repeat(current[:, -1:, ..., index : index + 1], width, axis=1))
                    appended = tf.concat(columns, axis=-1)
                else:
                    tf.debugging.assert_equal(
                        tf.shape(current)[-1],
                        tf.shape(appended)[-1],
                        message=f"{field_name} layouts differ. Provide feature_names metadata",
                    )
            return roll(current, appended)

        past_observed_mask = None
        if self.past_observed_mask is not None:
            generated_mask = tf.zeros_like(value, dtype=self.past_observed_mask.dtype)
            past_observed_mask = roll(self.past_observed_mask, generated_mask)

        padding_mask = None
        if self.padding_mask is not None:
            generated_padding = tf.ones([self.batch_size, width], dtype=self.padding_mask.dtype)
            padding_mask = roll(self.padding_mask, generated_padding)

        return replace(
            self,
            past_values=roll(self.past_values, value),
            past_time_features=advance_features(
                self.past_time_features,
                self.future_time_features,
                "past_real",
                "future_real",
                "real features",
            ),
            past_categorical_features=advance_features(
                self.past_categorical_features,
                self.future_categorical_features,
                "past_categorical",
                "future_categorical",
                "categorical features",
            ),
            past_observed_mask=past_observed_mask,
            padding_mask=padding_mask,
            future_values=None,
            future_observed_mask=None,
            labels=None,
            future_time_features=(
                None if self.future_time_features is None else self.future_time_features[:, width:, ...]
            ),
            future_categorical_features=(
                None if self.future_categorical_features is None else self.future_categorical_features[:, width:, ...]
            ),
        )
