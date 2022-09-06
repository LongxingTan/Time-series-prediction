# encoding=utf-8
"""
tfts package for time series prediction with TensorFlow.
"""

from tfts.model import AutoModel, build_tfts_model
from tfts.config import AutoConfig


__all__ = [
    "AutoModel",
    "build_tfts_model",
    "AutoConfig"
]

__version__ = "0.0.1"
