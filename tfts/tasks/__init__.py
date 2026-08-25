"""Task-specific handlers and pipelines for time series prediction."""

from .auto_task import (
    AnomalyHead,
    ClassificationHead,
    DistributionHead,
    GaussianHead,
    PredictionHead,
    QuantileHead,
    SegmentationHead,
)
from .base import BaseHead, BaseTask, ModelOutput
from .pipeline import Pipeline

__all__ = [
    "AnomalyHead",
    "BaseHead",
    "BaseTask",
    "ClassificationHead",
    "DistributionHead",
    "GaussianHead",
    "ModelOutput",
    "Pipeline",
    "PredictionHead",
    "QuantileHead",
    "SegmentationHead",
]
