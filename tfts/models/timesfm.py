"""
`A decoder-only foundation model for time-series forecasting
<https://arxiv.org/abs/2310.10688>`_
"""

import logging

import tensorflow as tf

from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


class TimesFmConfig(BaseConfig):
    model_type: str = "timesfm"

    def __init__(self):
        super().__init__()
