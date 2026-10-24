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


@register_transform("fourier")
class FourierTransform(FeatureTransform):
    """Generate sine/cosine pairs for a calendar period."""

    def __init__(self, attribute: str, period: int, order: int = 1, tags=("calendar", "fourier")):
        if period <= 0 or order <= 0:
            raise ValueError("period and order must be positive")
        self.attribute = attribute
        self.period = period
        self.order = order
        self.tags = frozenset(tags)

    def transform(self, frame, schema):
        result = frame.copy()
        times = pd.to_datetime(result[schema.time_col])
        if not hasattr(times.dt, self.attribute):
            raise ValueError(f"unsupported datetime attribute {self.attribute!r}")
        values = getattr(times.dt, self.attribute).astype(float)
        specs = []
        for harmonic in range(1, self.order + 1):
            angle = 2 * np.pi * harmonic * values / self.period
            for function, generated in (("sin", np.sin(angle)), ("cos", np.cos(angle))):
                name = f"{schema.time_col}_{self.attribute}_{function}_{harmonic}"
                result[name] = generated
                specs.append(
                    FeatureSpec(
                        name,
                        FeatureRole.KNOWN_FUTURE,
                        FeatureDType.REAL,
                        source=schema.time_col,
                        transform="fourier",
                        parameters={"attribute": self.attribute, "period": self.period, "harmonic": harmonic},
                        tags=self.tags,
                    )
                )
        return result, specs
