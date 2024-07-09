"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

import tensorflow as tf

from .base import BaseConfig, BaseModel

config = {
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class TFTransformer(object):
    """Temporal fusion transformer model"""

    def __init__(self, predict_sequence_length=3, custom_model_config=None):
        if custom_model_config:
            config.update(custom_model_config)
        self.config = config
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        return
