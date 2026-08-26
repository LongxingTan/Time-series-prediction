"""tfts models"""

from .auto_config import AutoConfig
from .auto_model import (
    AutoBackbone,
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForAnomalyDetection,
    AutoModelForClassification,
    AutoModelForForecasting,
    AutoModelForImputation,
    AutoModelForPrediction,
    AutoModelForQuantile,
    AutoModelForTimeSeriesClassification,
)
from .base import BaseConfig, BaseModel, CommonConfig
from .registry import get_model_capabilities, get_model_info, list_models, list_supported_tasks, register_model
