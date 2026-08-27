"""Serializable controls for forecast generation."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ForecastGenerationConfig:
    prediction_length: Optional[int] = None
    strategy: str = "auto"
    sampler: str = "auto"
    num_samples: int = 100
    aggregation: str = "mean"
    return_samples: bool = False
    quantiles: Tuple[float, ...] = ()
    seed: Optional[int] = None

    def __post_init__(self):
        if self.prediction_length is not None and self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.strategy not in {"auto", "direct", "recursive", "autoregressive", "diffusion"}:
            raise ValueError("Unknown generation strategy %r" % self.strategy)
        if self.sampler not in {"auto", "mean", "sample"}:
            raise ValueError("Unknown generation sampler %r" % self.sampler)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.aggregation not in {"mean", "median", "none"}:
            raise ValueError("Unknown sample aggregation %r" % self.aggregation)

    @classmethod
    def from_args(cls, value=None, **overrides):
        if value is None:
            values: Dict[str, Any] = {}
        elif isinstance(value, cls):
            values = asdict(value)
        elif isinstance(value, dict):
            values = dict(value)
        else:
            raise TypeError("generation_config must be a mapping, ForecastGenerationConfig, or None")
        values.update(overrides)
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError("Unknown generation config fields: %s" % sorted(unknown))
        return cls(**values)

    def to_dict(self):
        return asdict(self)
