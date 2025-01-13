"""
`Are Transformers Effective for Time Series Forecasting?
<https://arxiv.org/abs/2205.13504>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from .base import BaseConfig, BaseModel


class DLinearConfig(BaseConfig):
    model_type: str = "dlinear"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate


class DLinear(BaseModel):
    """DLinear Network"""

    def __init__(self, predict_sequence_length: int = 1, config=DLinearConfig()):
        super(DLinear, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        # Create a list of dense layers for the model
        self.dense_layers = []
        for _ in range(self.config.num_layers):
            self.dense_layers.append(Dense(units=self.config.hidden_size, activation="relu"))
            self.dense_layers.append(Dropout(self.config.dropout_rate))

        self.output_layer = Dense(units=predict_sequence_length)

    def __call__(self, inputs: tf.Tensor, return_dict: Optional[bool] = None):
        """DLinear model forward pass

        Parameters
        ----------
        inputs : tf.Tensor
            Input time-series data.

        Returns
        -------
        tf.Tensor
            Forecasted values.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        outputs = self.output_layer(x)

        return outputs
