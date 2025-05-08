"""
`Are Transformers Effective for Time Series Forecasting?
<https://arxiv.org/abs/2205.13504>`_
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda

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
        self.activation: Optional[str] = None
        self.initializer: str = "glorot_uniform"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive and odd, got {self.kernel_size}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


class DLinear(BaseModel):
    """DLinear Network for Time Series Forecasting."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[DLinearConfig] = None):
        super(DLinear, self).__init__()
        self.config = config or DLinearConfig()
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
        """DLinear model forward pass.

        Args:
            inputs: Input time-series data of shape [batch_size, seq_len, channels].
            training: Whether the model is in training mode.
            output_hidden_states: Whether to return intermediate hidden states.
            return_dict: Whether to return outputs as a dictionary.

        Returns:
            If return_dict is False:
                Forecasted values tensor of shape [batch_size, predict_sequence_length, input_dim]
            If return_dict is True:
                Dictionary containing:
                - 'predictions': Forecasted values
                - 'seasonal_component': Seasonal component (if output_hidden_states is True)
                - 'trend_component': Trend component (if output_hidden_states is True)
        """
        # Decompose the input into trend and seasonal components
        seasonal, trend = self.decomposition(inputs)

        seasonal = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(seasonal)
        trend = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(trend)

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
        output = Lambda(lambda t: tf.transpose(t, [0, 2, 1]))(output)
        output = self.project(output)
        return output
