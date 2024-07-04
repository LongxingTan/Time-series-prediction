"""tfts package for time series prediction with TensorFlow"""

from tfts.datasets.get_data import get_data
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
)
from tfts.trainer import KerasTrainer, Trainer
from tfts.tuner import AutoTuner

__all__ = [
    "AutoModel",
    "AutoModelForPrediction",
    "AutoModelForClassification",
    "AutoModelForSegmentation",
    "AutoModelForAnomaly" "AutoConfig",
    "AutoTuner",
    "Trainer",
    "KerasTrainer",
    "get_data",
]

__version__ = "0.0.0"
