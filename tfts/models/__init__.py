"""tfts models"""

from .auto_config import AutoConfig
from .auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForQuantile,
    AutoModelForSegmentation,
    AutoModelForUncertainty,
)
from .base import BaseConfig, BaseModel, CommonConfig
from .registry import get_model_info, list_models, register_model
