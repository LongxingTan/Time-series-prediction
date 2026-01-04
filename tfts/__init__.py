"""tfts package for time series prediction with TensorFlow"""

from tfts.data import TimeSeriesSequence, get_data
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import (
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
    AutoModelForUncertainty,
)
from tfts.tasks.pipeline import Pipeline
from tfts.trainer import KerasTrainer, Trainer, set_seed
from tfts.training_args import TrainingArguments

__all__ = [
    "AutoModel",
    "AutoModelForPrediction",
    "AutoModelForClassification",
    "AutoModelForSegmentation",
    "AutoModelForAnomaly",
    "AutoModelForUncertainty",
    "AutoConfig",
    "Trainer",
    "KerasTrainer",
    "TrainingArguments",
    "set_seed" "Pipeline",
    "get_data",
    "TimeSeriesSequence",
]

__version__ = "0.0.0"
