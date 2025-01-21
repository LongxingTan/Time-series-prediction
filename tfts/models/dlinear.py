"""
`Are Transformers Effective for Time Series Forecasting?
<https://arxiv.org/abs/2205.13504>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from tfts.layers.autoformer_layer import SeriesDecomp

from .base import BaseConfig, BaseModel


class DLinearConfig(BaseConfig):
    model_type: str = "dlinear"

    def __init__(
        self,
        kernel_size: int = 25,
        channels: int = 3,
        individual: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels  # number of input features
        self.individual = individual


class DLinear(BaseModel):
    """DLinear Network"""

    def __init__(self, predict_sequence_length: int = 1, config=DLinearConfig()):
        super(DLinear, self).__init__()
        self.config = config
        self.predict_sequence_length = predict_sequence_length

        self.decomposition = SeriesDecomp(self.config.kernel_size)
        if self.config.individual:
            self.linear_seasonal = [Dense(self.predict_sequence_length) for _ in range(self.config.channels)]
            self.linear_trend = [Dense(self.predict_sequence_length) for _ in range(self.config.channels)]
        else:
            self.linear_seasonal = Dense(self.predict_sequence_length)
            self.linear_trend = Dense(self.predict_sequence_length)
        self.project = Dense(1)

    def __call__(
        self, inputs: tf.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
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
        # Decompose the input into trend and seasonal components
        seasonal, trend = self.decomposition(inputs)

        seasonal = tf.transpose(seasonal, [0, 2, 1])
        trend = tf.transpose(trend, [0, 2, 1])

        if self.config.individual:
            seasonal_output = []
            trend_output = []
            for i in range(self.config.channels):
                seasonal_output.append(self.linear_seasonal[i](seasonal[:, i, :]))
                trend_output.append(self.linear_trend[i](trend[:, i, :]))
            seasonal_output = tf.stack(seasonal_output, axis=1)
            trend_output = tf.stack(trend_output, axis=1)
        else:
            seasonal_output = self.linear_seasonal(seasonal)
            trend_output = self.linear_trend(trend)

        output = seasonal_output + trend_output
        output = tf.transpose(output, [0, 2, 1])
        output = self.project(output)
        return output
