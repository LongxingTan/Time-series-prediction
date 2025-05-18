"""Layer for :py:class:`~tfts.models.nbeats`"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Layer


class GenericBlock(tf.keras.layers.Layer):
    """Generic block that learns arbitrary patterns in time series data.

    This block uses a stack of fully connected layers to learn any pattern in the input data.
    It outputs both backcast (reconstruction of input) and forecast (prediction) components.

    Parameters
    ----------
    train_sequence_length : int
        Length of the input sequence
    predict_sequence_length : int
        Length of the prediction sequence
    hidden_size : int
        Number of units in the hidden layers
    n_block_layers : int, optional
        Number of fully connected layers in the block, by default 4
    """

    def __init__(
        self, train_sequence_length: int, predict_sequence_length: int, hidden_size: int, n_block_layers: int = 4
    ):
        super(GenericBlock, self).__init__()
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_block_layers = n_block_layers

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the layer's weights.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape of the input tensor
        """
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(self.train_sequence_length + self.predict_sequence_length, use_bias=False, activation=None)
        super(GenericBlock, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the output of the Generic Block.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape (batch_size, train_sequence_length, input_size)

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple of two tensors:
            - backcast: Shape (batch_size, train_sequence_length, output_size)
            - forecast: Shape (batch_size, predict_sequence_length, output_size)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)
        return x[:, : self.train_sequence_length], x[:, -self.predict_sequence_length :]


class TrendBlock(tf.keras.layers.Layer):
    """Trend block that learns trend patterns using polynomial basis functions.

    This block uses polynomial basis functions to model trends in the time series data.
    It outputs both backcast (reconstruction of input) and forecast (prediction) components.

    Parameters
    ----------
    train_sequence_length : int
        Length of the input sequence
    predict_sequence_length : int
        Length of the prediction sequence
    hidden_size : int
        Number of units in the hidden layers
    n_block_layers : int, optional
        Number of fully connected layers in the block, by default 4
    polynomial_term : int, optional
        Degree of the polynomial basis functions, by default 2
    """

    def __init__(
        self,
        train_sequence_length: int,
        predict_sequence_length: int,
        hidden_size: int,
        n_block_layers: int = 4,
        polynomial_term: int = 2,
    ):
        super().__init__()

        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_block_layers = n_block_layers
        self.polynomial_size = polynomial_term + 1

        self.forecast_time = tf.concat(
            [
                tf.math.pow(tf.range(predict_sequence_length, dtype=tf.float32) / predict_sequence_length, i)[None, :]
                for i in range(self.polynomial_size)
            ],
            axis=0,
        )
        self.backcast_time = tf.concat(
            [
                tf.math.pow(
                    tf.range(train_sequence_length, dtype=tf.float32) / tf.cast(train_sequence_length, tf.float32), i
                )[None, :]
                for i in range(self.polynomial_size)
            ],
            axis=0,
        )

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the layer's weights.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape of the input tensor
        """

        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(2 * self.polynomial_size, use_bias=False, activation=None)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the output of the Trend Block.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape (batch_size, train_sequence_length, input_size)

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple of two tensors:
            - backcast: Shape (batch_size, train_sequence_length, output_size)
            - forecast: Shape (batch_size, predict_sequence_length, output_size)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)
        backcast = tf.einsum("bp,pt->bt", x[:, self.polynomial_size :], self.backcast_time)
        forecast = tf.einsum("bp,pt->bt", x[:, : self.polynomial_size], self.forecast_time)
        return backcast, forecast

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.train_sequence_length), (input_shape[0], self.predict_sequence_length)]


class SeasonalityBlock(tf.keras.layers.Layer):
    """Seasonality block"""

    def __init__(self, train_sequence_length, predict_sequence_length, hidden_size, n_block_layers=4, num_harmonics=1):
        super().__init__()
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_block_layers = n_block_layers
        self.num_harmonics = num_harmonics
        self.theta_size = 4 * int(np.ceil(num_harmonics / 2 * predict_sequence_length) - (num_harmonics - 1))

        self.frequency = tf.concat(
            [
                tf.zeros(1, dtype=tf.float32),
                tf.range(num_harmonics, num_harmonics / 2 * predict_sequence_length, dtype=tf.float32),
            ],
            axis=0,
        )
        self.backcast_grid = (
            -2
            * np.pi
            * (
                tf.range(self.train_sequence_length, dtype=tf.float32)[:, None]
                / tf.cast(train_sequence_length, tf.float32)
            )
            * self.frequency
        )
        self.forecast_grid = (
            2
            * np.pi
            * (
                tf.range(predict_sequence_length, dtype=tf.float32)[:, None]
                / tf.cast(predict_sequence_length, tf.float32)
            )
            * self.frequency
        )
        self.backcast_cos_template = tf.transpose(tf.cos(self.backcast_grid))
        self.backcast_sin_template = tf.transpose(tf.sin(self.backcast_grid))
        self.forecast_cos_template = tf.transpose(tf.cos(self.forecast_grid))
        self.forecast_sin_template = tf.transpose(tf.sin(self.forecast_grid))

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(self.theta_size, use_bias=False, activation=None)

    def call(self, inputs):
        """Compute the output of the Seasonality Block.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of shape (batch_size, train_sequence_length, input_size)

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple of two tensors:
            - backcast: Shape (batch_size, train_sequence_length, output_size)
            - forecast: Shape (batch_size, predict_sequence_length, output_size)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)

        config_per_harmonic = self.theta_size // 4

        backcast_harmonics_cos = tf.einsum(
            "bp,pt->bt", inputs[:, 2 * config_per_harmonic : 3 * config_per_harmonic], self.backcast_cos_template
        )
        backcast_harmonics_sin = tf.einsum("bp,pt->bt", x[:, 3 * config_per_harmonic :], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = tf.einsum("bp,pt->bt", x[:, :config_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = tf.einsum(
            "bp,pt->bt", x[:, config_per_harmonic : 2 * config_per_harmonic], self.forecast_sin_template
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class BackcastMinusLayer(tf.keras.layers.Layer):
    """Layer for computing backcast minus operation"""

    def call(self, inputs):
        backcast, b = inputs
        return backcast - b

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return shape of first input


class ForecastPlusLayer(tf.keras.layers.Layer):
    """Layer for computing forecast plus operation"""

    def call(self, inputs):
        forecast, f = inputs
        return forecast + f

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return shape of first input


class ZerosLayer(tf.keras.layers.Layer):
    """Layer for creating zeros tensor with proper shape"""

    def __init__(self, predict_length, **kwargs):
        super(ZerosLayer, self).__init__(**kwargs)
        self.predict_length = predict_length

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.zeros([batch_size, self.predict_length], dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.predict_length)
