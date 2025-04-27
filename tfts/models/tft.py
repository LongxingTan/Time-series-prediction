"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

from typing import Optional

import tensorflow as tf

from .base import BaseConfig, BaseModel


class TFTransformerConfig(BaseConfig):
    model_type: str = "tft"

    def __init__(self):
        super(TFTransformerConfig, self).__init__()


class TFTransformer(BaseModel):
    """Temporal fusion transformer model"""

    def __init__(self, predict_sequence_length=1, config: Optional[TFTransformerConfig] = None):
        super(TFTransformer, self).__init__()
        self.config = config or TFTransformerConfig()
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None):
        """_summary_

        Parameters
        ----------
        x : tf.Tensor
            _description_
        """
        return
