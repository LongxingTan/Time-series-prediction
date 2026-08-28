"""Stable contracts shared by TFTS models, tasks, and generation."""

from .batch import TimeSeriesBatch
from .capabilities import BackboneCapabilities, ForecastMode, InputLayout, ModelInputSpec, OutputPort
from .outputs import (
    AnomalyDetectionOutput,
    BackboneOutput,
    ClassificationOutput,
    ForecastOutput,
    ImputationOutput,
    ModelOutput,
)
from .task import (
    AnomalyDetectionTaskConfig,
    ClassificationTaskConfig,
    ForecastTaskConfig,
    ImputationTaskConfig,
    TaskConfig,
    TaskType,
)

__all__ = [
    "AnomalyDetectionOutput",
    "AnomalyDetectionTaskConfig",
    "BackboneCapabilities",
    "BackboneOutput",
    "ClassificationOutput",
    "ClassificationTaskConfig",
    "ForecastMode",
    "ForecastOutput",
    "ForecastTaskConfig",
    "ImputationOutput",
    "ImputationTaskConfig",
    "InputLayout",
    "ModelOutput",
    "ModelInputSpec",
    "OutputPort",
    "TaskConfig",
    "TaskType",
    "TimeSeriesBatch",
]
