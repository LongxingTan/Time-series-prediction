# encoding=utf-8
"""
tfts package for time series prediction with TensorFlow.
"""

from tfts.config import AutoConfig
from tfts.model import AutoModel, build_tfts_model
from tfts.trainer import Trainer, KerasTrainer
from tfts.data import load


__all__ = [
    "AutoModel",
    "build_tfts_model",
    "AutoConfig"
    "Trainer",
    "KerasTrainer",
    "load"
]

__version__ = "0.0.2"
