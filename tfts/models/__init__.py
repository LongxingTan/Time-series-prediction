"""tfts models"""

from .auto_config import AutoConfig
from .auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
    AutoModelForUncertainty,
)
from .base import BaseConfig, BaseModel
from .registry import list_models
