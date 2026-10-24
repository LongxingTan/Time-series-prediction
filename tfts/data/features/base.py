"""Composable, backend-neutral feature transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tfts.data.schema import FeatureDType, FeatureManifest, FeatureRole, FeatureSpec, Schema


@dataclass(frozen=True)
class PreparedTimeSeries:
    """A sorted feature frame together with its fitted semantic manifest."""

    frame: pd.DataFrame
    manifest: FeatureManifest

    @property
    def schema(self) -> Schema:
        return self.manifest.schema


class FeatureTransform(ABC):
    """Base contract for deterministic feature transformations."""

    @property
    def required_history(self) -> int:
        return 0

    def fit(self, frame: pd.DataFrame, schema: Schema) -> Mapping[str, Any]:
        return {}

    def output_required_history(self, spec: FeatureSpec, source_history: Mapping[str, int]) -> int:
        """Return warm-up rows required by one output feature."""
        return self.required_history

    def state_dict(self):
        return {}

    def load_state(self, state):
        if state:
            raise ValueError(f"{type(self).__name__} has no fitted state")

    def to_dict(self):
        """Serialize registered constructor choices separately from fitted state."""
        import inspect

        from tfts.registry import registrations

        names = [name for name, entry in registrations("transform").items() if entry.factory is type(self)]
        if len(names) != 1:
            raise ValueError("Feature transforms must have one registered name to serialize")
        config = {}
        for name, parameter in inspect.signature(type(self).__init__).parameters.items():
            if name == "self":
                continue
            if not hasattr(self, name):
                raise ValueError(f"Transform constructor option {name!r} must be stored")
            value = getattr(self, name)
            config[name] = sorted(value) if isinstance(value, (set, frozenset)) else value
        return {"name": names[0], "config": config, "state": self.state_dict()}

    @classmethod
    def from_dict(cls, values):
        from tfts.registry import resolve

        transform = resolve("transform", values["name"]).factory(**values["config"])
        if not isinstance(transform, FeatureTransform):
            raise TypeError("Registered transform must implement FeatureTransform")
        transform.load_state(values["state"])
        return transform

    @abstractmethod
    def transform(self, frame: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, Sequence[FeatureSpec]]:
        """Return a copied frame and the specs for newly generated columns."""


class FeaturePipeline:
    """Fit and apply an ordered collection of feature transformations."""

    def __init__(self, transforms: Optional[Sequence[FeatureTransform]] = None):
        self.transforms = tuple(transforms or ())
        self.manifest_: Optional[FeatureManifest] = None
        self.input_schema_: Optional[Schema] = None

    def fit(self, frame: pd.DataFrame, schema: Schema) -> "FeaturePipeline":
        prepared, states, final_schema, required_history = self._run(frame, schema, fit=True)
        del prepared
        self.manifest_ = FeatureManifest(
            final_schema,
            required_history=required_history,
            fitted_state=states,
        )
        self.input_schema_ = schema
        return self

    def transform(self, frame: pd.DataFrame) -> PreparedTimeSeries:
        if self.manifest_ is None:
            raise RuntimeError("FeaturePipeline is not fitted; call fit() or fit_transform() first")
        if self.input_schema_ is None:
            raise RuntimeError("FeaturePipeline input schema is unavailable")
        prepared, _, final_schema, _ = self._run(frame, self.input_schema_, fit=False)
        if final_schema.to_dict() != self.manifest_.schema.to_dict():
            raise RuntimeError("feature pipeline output schema changed after fitting")
        return PreparedTimeSeries(prepared, self.manifest_)

    def fit_transform(self, frame: pd.DataFrame, schema: Schema) -> PreparedTimeSeries:
        prepared, states, final_schema, required_history = self._run(frame, schema, fit=True)
        self.manifest_ = FeatureManifest(
            final_schema,
            required_history=required_history,
            fitted_state=states,
        )
        self.input_schema_ = schema
        return PreparedTimeSeries(prepared, self.manifest_)

    def to_dict(self):
        if self.manifest_ is None:
            raise RuntimeError("FeaturePipeline is not fitted")
        return {
            "transforms": [transform.to_dict() for transform in self.transforms],
            "input_schema": self.input_schema_.to_dict(),
            "manifest": self.manifest_.to_dict(),
        }

    @classmethod
    def from_dict(cls, values):
        pipeline = cls([FeatureTransform.from_dict(value) for value in values["transforms"]])
        pipeline.input_schema_ = Schema.from_dict(values["input_schema"])
        pipeline.manifest_ = FeatureManifest.from_dict(values["manifest"])
        return pipeline

    def _run(self, frame, schema, fit):
        schema.validate_frame(frame)
        sort_columns = list(schema.group_cols) + [schema.time_col]
        result = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True).copy()
        current_schema = schema
        histories = {spec.name: 0 for spec in schema.features}
        states: Dict[str, Any] = {}
        for index, transform in enumerate(self.transforms):
            if fit:
                states[f"{index}:{type(transform).__name__}"] = dict(transform.fit(result, current_schema))
            result, specs = transform.transform(result, current_schema)
            histories.update({spec.name: transform.output_required_history(spec, histories) for spec in specs})
            current_schema = current_schema.with_features(specs)
        current_schema.validate_frame(result)
        self._validate_static_features(result, current_schema)
        return result, states, current_schema, max(histories.values(), default=0)

    @staticmethod
    def _validate_static_features(frame, schema):
        static = [spec.name for spec in schema.features if spec.role == FeatureRole.STATIC]
        if not static:
            return
        if schema.group_cols:
            counts = frame.groupby(list(schema.group_cols), observed=True, sort=False)[static].nunique(dropna=False)
            invalid = [column for column in static if (counts[column] > 1).any()]
        else:
            invalid = [column for column in static if frame[column].nunique(dropna=False) > 1]
        if invalid:
            raise ValueError(f"static features vary within a time series: {invalid}")


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _json_scalar(value):
    value = _python_scalar(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
