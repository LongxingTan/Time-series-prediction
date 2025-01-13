"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

import tensorflow as tf

from .base import BaseConfig, BaseModel


class TFTransformerConfig(BaseConfig):
    model_type = "tft"

    def __init__(self):
        super(TFTransformerConfig, self).__init__()


class TFTransformer(BaseModel):
    """Temporal fusion transformer model"""

    def __init__(self, predict_sequence_length=1, config=None):
        super(TFTransformer, self).__init__()
        if config is None:
            config = TFTransformerConfig()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x: tf.Tensor):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        return
