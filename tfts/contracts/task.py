"""Serializable task configuration, separate from backbone architecture."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class TaskType(str, Enum):
    FORECASTING = "forecasting"
    IMPUTATION = "imputation"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"

    @classmethod
    def normalize(cls, value):
        if isinstance(value, cls):
            return value
        normalized = str(value).lower().replace("-", "_")
        return cls(normalized)


@dataclass(frozen=True)
class TaskConfig:
    task: TaskType

    def __post_init__(self):
        object.__setattr__(self, "task", TaskType.normalize(self.task))

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["task"] = self.task.value
        return values


@dataclass(frozen=True)
class ForecastTaskConfig(TaskConfig):
    task: TaskType = TaskType.FORECASTING
    prediction_length: int = 1
    target_dim: int = 1
    head: str = "auto"
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    residual: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        if self.prediction_length <= 0 or self.target_dim <= 0:
            raise ValueError("prediction_length and target_dim must be positive")
        if self.head not in {"auto", "point", "quantile", "distribution", "native"}:
            raise ValueError("Unknown forecast head %r" % self.head)
        if self.head == "quantile":
            if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
                raise ValueError("quantiles must be unique and sorted")
            if any(q <= 0 or q >= 1 for q in self.quantiles):
                raise ValueError("quantiles must lie strictly between 0 and 1")
        if self.residual not in {None, "last_value", "mean", "last_window"}:
            raise ValueError("Unknown residual %r" % self.residual)


@dataclass(frozen=True)
class ClassificationTaskConfig(TaskConfig):
    task: TaskType = TaskType.CLASSIFICATION
    num_labels: int = 2
    hidden_units: Tuple[int, ...] = (128,)
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "hidden_units", tuple(int(units) for units in self.hidden_units))
        if self.num_labels <= 1:
            raise ValueError("num_labels must be greater than one")
        if any(units <= 0 for units in self.hidden_units):
            raise ValueError("hidden_units must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass(frozen=True)
class ImputationTaskConfig(TaskConfig):
    task: TaskType = TaskType.IMPUTATION
    target_dim: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")


@dataclass(frozen=True)
class AnomalyDetectionTaskConfig(TaskConfig):
    task: TaskType = TaskType.ANOMALY_DETECTION
    target_dim: int = 1
    scorer: str = "squared_error"
    threshold_quantile: float = 0.99

    def __post_init__(self):
        super().__post_init__()
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if self.scorer not in {"squared_error", "absolute_error"}:
            raise ValueError("Unknown anomaly scorer %r" % self.scorer)
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must lie strictly between 0 and 1")
