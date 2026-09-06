"""Serializable generation configuration."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Tuple

from tfts.registry import ComponentSpec


def _spec(value):
    return ComponentSpec.from_value(value)


@dataclass(frozen=True)
class GenerationConfig:
    horizon: Optional[int] = None
    mode: str = "auto"
    processors: Tuple[ComponentSpec, ...] = (ComponentSpec("auto"),)
    trajectory: Tuple[ComponentSpec, ...] = (ComponentSpec("inverse_scale"),)
    num_samples: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.mode not in {"auto", "direct", "recursive", "native"}:
            raise ValueError(f"Unknown generation mode {self.mode!r}")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        object.__setattr__(self, "processors", tuple(_spec(value) for value in self.processors))
        object.__setattr__(self, "trajectory", tuple(_spec(value) for value in self.trajectory))

    def to_dict(self):
        values = asdict(self)
        values["processors"] = [value.to_dict() for value in self.processors]
        values["trajectory"] = [value.to_dict() for value in self.trajectory]
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        return cls(**dict(values))

    @classmethod
    def from_value(cls, value=None, **overrides):
        if value is None:
            values = {}
        elif isinstance(value, cls):
            values = value.to_dict()
        elif isinstance(value, Mapping):
            values = dict(value)
        else:
            raise TypeError("config must be a mapping, GenerationConfig, or None")
        values.update(overrides)
        return cls.from_dict(values)
