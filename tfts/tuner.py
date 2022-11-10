"""tfts auto tuner"""

import numpy as np
import optuna

from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel

__all__ = ["AutoTuner"]


class AutoTuner(object):
    """Auto tune parameters by optuna"""

    def __init__(self, use_model: str):
        self.use_model = use_model

    def generate_parameter(self):
        pass

    def run(self, config, direction="maximize"):
        pass
