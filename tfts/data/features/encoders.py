"""Composable, backend-neutral feature transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tfts.data.features.base import FeatureTransform, _json_scalar, _python_scalar
from tfts.data.schema import FeatureDType, FeatureManifest, FeatureRole, FeatureSpec, Schema
from tfts.registry import register_transform


@register_transform("categoricalencoder")
class CategoricalEncoderTransform(FeatureTransform):
    """Fit stable integer codes, reserving one value for missing/unseen data."""

    def __init__(self, columns, suffix: str = "_encoded", unknown_value: int = 0, tags=("encoded",)):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        if not self.columns:
            raise ValueError("columns must contain at least one feature")
        if not suffix:
            raise ValueError("suffix must be non-empty")
        self.suffix = suffix
        self.unknown_value = int(unknown_value)
        if self.unknown_value < 0:
            raise ValueError("unknown_value must be non-negative for embedding compatibility")
        self.tags = frozenset(tags)
        self.categories_: Dict[str, Tuple[Any, ...]] = {}

    def fit(self, frame, schema):
        categories = {}
        for column in self.columns:
            spec = schema.get(column)
            if spec.role == FeatureRole.TARGET:
                raise ValueError("CategoricalEncoderTransform cannot encode a target column")
            values = frame[column].dropna().drop_duplicates().tolist()
            categories[column] = tuple(_python_scalar(value) for value in values)
        self.categories_ = categories
        return {column: [_json_scalar(value) for value in values] for column, values in categories.items()}

    def state_dict(self):
        return {name: [_json_scalar(value) for value in values] for name, values in self.categories_.items()}

    def load_state(self, state):
        if set(state) != set(self.columns):
            raise ValueError("Category state must match configured columns")
        self.categories_ = {name: tuple(values) for name, values in state.items()}

    def transform(self, frame, schema):
        if set(self.categories_) != set(self.columns):
            raise RuntimeError("CategoricalEncoderTransform is not fitted")
        result = frame.copy()
        specs = []
        for column in self.columns:
            source = schema.get(column)
            available_codes = (code for code in range(len(self.categories_[column]) + 1) if code != self.unknown_value)
            mapping = {value: code for value, code in zip(self.categories_[column], available_codes)}
            cardinality = max([self.unknown_value, *mapping.values()]) + 1
            name = f"{column}{self.suffix}"
            result[name] = result[column].map(mapping).fillna(self.unknown_value).astype("int32")
            specs.append(
                FeatureSpec(
                    name,
                    source.role,
                    FeatureDType.CATEGORICAL,
                    source=column,
                    transform="categorical_encode",
                    parameters={"unknown_value": self.unknown_value, "cardinality": cardinality},
                    tags=source.tags | self.tags,
                )
            )
        return result, specs
