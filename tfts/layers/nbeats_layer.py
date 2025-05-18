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


class TimeBasesLayer(Layer):
    """Layer to create polynomial or Fourier bases for time series."""

    def __init__(self, sequence_length, polynomial_size, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.polynomial_size = polynomial_size

    def build(self, input_shape):
        # Pre-compute time bases to avoid TF function calls during forward pass
        bases = []
        for i in range(self.polynomial_size):
            # Create normalized time indices (0 to 1)
            time_range = tf.range(self.sequence_length, dtype=tf.float32) / self.sequence_length
            # Compute power for this basis
            power_basis = tf.pow(time_range, tf.cast(i, tf.float32))
            bases.append(power_basis)

        # Shape: (polynomial_size, sequence_length)
        self.bases = tf.stack(bases)
        super().build(input_shape)

    def call(self, inputs):
        # inputs are ignored, we just return the bases
        batch_size = tf.shape(inputs)[0]
        # Replicate bases for each batch item
        # We're returning shape (batch_size, polynomial_size, sequence_length)
        return tf.tile(self.bases[None, :, :], [batch_size, 1, 1])


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

        # Create layers for the time bases
        self.backcast_bases_layer = TimeBasesLayer(train_sequence_length, self.polynomial_size)
        self.forecast_bases_layer = TimeBasesLayer(predict_sequence_length, self.polynomial_size)

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

        # Get time bases for backcast and forecast
        backcast_time = self.backcast_bases_layer(x)  # Shape: (batch_size, polynomial_size, train_sequence_length)
        forecast_time = self.forecast_bases_layer(x)  # Shape: (batch_size, polynomial_size, predict_sequence_length)

        # Split theta parameters for backcast and forecast
        backcast_params = x[:, self.polynomial_size :]  # Shape: (batch_size, polynomial_size)
        forecast_params = x[:, : self.polynomial_size]  # Shape: (batch_size, polynomial_size)

        # Reshape params for batch matmul
        backcast_params = tf.expand_dims(backcast_params, axis=-1)  # (batch_size, polynomial_size, 1)
        forecast_params = tf.expand_dims(forecast_params, axis=-1)  # (batch_size, polynomial_size, 1)

        # Compute backcast and forecast
        backcast = tf.matmul(backcast_time, backcast_params)  # (batch_size, train_sequence_length, 1)
        forecast = tf.matmul(forecast_time, forecast_params)  # (batch_size, predict_sequence_length, 1)

        # Reshape to remove last dimension
        backcast = tf.squeeze(backcast, axis=-1)  # (batch_size, train_sequence_length)
        forecast = tf.squeeze(forecast, axis=-1)  # (batch_size, predict_sequence_length)

        return backcast, forecast


class FourierBasesLayer(Layer):
    """Layer to create Fourier bases for time series."""

    def __init__(self, sequence_length, num_harmonics, is_backcast=True, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.num_harmonics = num_harmonics
        self.is_backcast = is_backcast

    def build(self, input_shape):
        # Create normalized time grid
        t = tf.range(self.sequence_length, dtype=tf.float32) / self.sequence_length

        # Generate frequencies
        frequencies = []
        if self.is_backcast:
            factor = -2.0 * np.pi  # Negative for backcast
        else:
            factor = 2.0 * np.pi  # Positive for forecast

        # First frequency is zero (DC component)
        frequencies.append(tf.zeros_like(t))

        # Add harmonic frequencies
        for h in range(1, self.num_harmonics + 1):
            frequencies.append(h * tf.ones_like(t))

        # Stack frequencies
        frequencies = tf.stack(frequencies)  # (num_harmonics+1, sequence_length)

        # Compute time grid * frequencies
        self.grid = factor * tf.expand_dims(t, 0) * frequencies  # (num_harmonics+1, sequence_length)

        # Compute sin and cos templates
        self.cos_template = tf.cos(self.grid)  # (num_harmonics+1, sequence_length)
        self.sin_template = tf.sin(self.grid)  # (num_harmonics+1, sequence_length)

        super().build(input_shape)

    def call(self, inputs):
        # inputs are ignored, we just return the templates
        batch_size = tf.shape(inputs)[0]
        cos_bases = tf.tile(self.cos_template[None, :, :], [batch_size, 1, 1])
        sin_bases = tf.tile(self.sin_template[None, :, :], [batch_size, 1, 1])
        return cos_bases, sin_bases


class SeasonalityBlock(tf.keras.layers.Layer):
    """Seasonality block that learns seasonal patterns using Fourier basis functions.

    This block uses Fourier basis functions to model seasonal patterns in the time series data.
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
    num_harmonics : int, optional
        Number of harmonics to use in the Fourier basis, by default 1
    """

    def __init__(
        self,
        train_sequence_length: int,
        predict_sequence_length: int,
        hidden_size: int,
        n_block_layers: int = 4,
        num_harmonics: int = 1,
    ):
        super().__init__()
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_block_layers = n_block_layers
        self.num_harmonics = num_harmonics

        # Each harmonic has a sin and cos component for both backcast and forecast
        self.theta_size = 4 * (self.num_harmonics + 1)  # +1 for the DC component

        # Create fourier bases layers
        self.backcast_bases = FourierBasesLayer(train_sequence_length, num_harmonics, is_backcast=True)
        self.forecast_bases = FourierBasesLayer(predict_sequence_length, num_harmonics, is_backcast=False)

    def build(self, input_shape: Tuple[Optional[int], ...]):
        """Build the layer's weights.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape of the input tensor
        """
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(self.theta_size, use_bias=False, activation=None)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

        # Split theta parameters into 4 groups
        params_per_group = self.theta_size // 4
        forecast_cos_params = x[:, :params_per_group]  # (batch_size, params_per_group)
        forecast_sin_params = x[:, params_per_group : 2 * params_per_group]  # (batch_size, params_per_group)
        backcast_cos_params = x[:, 2 * params_per_group : 3 * params_per_group]  # (batch_size, params_per_group)
        backcast_sin_params = x[:, 3 * params_per_group :]  # (batch_size, params_per_group)

        # Get Fourier bases
        backcast_cos_bases, backcast_sin_bases = self.backcast_bases(x)
        forecast_cos_bases, forecast_sin_bases = self.forecast_bases(x)

        # Reshape params for batch matmul
        backcast_cos_params = tf.expand_dims(backcast_cos_params, axis=-1)  # (batch_size, params_per_group, 1)
        backcast_sin_params = tf.expand_dims(backcast_sin_params, axis=-1)  # (batch_size, params_per_group, 1)
        forecast_cos_params = tf.expand_dims(forecast_cos_params, axis=-1)  # (batch_size, params_per_group, 1)
        forecast_sin_params = tf.expand_dims(forecast_sin_params, axis=-1)  # (batch_size, params_per_group, 1)

        # Compute backcast and forecast components
        backcast_cos = tf.matmul(backcast_cos_bases, backcast_cos_params)  # (batch_size, train_sequence_length, 1)
        backcast_sin = tf.matmul(backcast_sin_bases, backcast_sin_params)  # (batch_size, train_sequence_length, 1)
        forecast_cos = tf.matmul(forecast_cos_bases, forecast_cos_params)  # (batch_size, predict_sequence_length, 1)
        forecast_sin = tf.matmul(forecast_sin_bases, forecast_sin_params)  # (batch_size, predict_sequence_length, 1)

        # Combine components and reshape
        backcast = tf.squeeze(backcast_cos + backcast_sin, axis=-1)  # (batch_size, train_sequence_length)
        forecast = tf.squeeze(forecast_cos + forecast_sin, axis=-1)  # (batch_size, predict_sequence_length)

        return backcast, forecast


class BackcastMinusLayer(tf.keras.layers.Layer):
    """Layer for computing backcast minus operation"""

    def call(self, inputs):
        backcast, b = inputs
        return backcast - b


class ForecastPlusLayer(tf.keras.layers.Layer):
    """Layer for computing forecast plus operation"""

    def call(self, inputs):
        forecast, f = inputs
        return forecast + f


class ShapeLayer(tf.keras.layers.Layer):
    """Layer for getting tensor shape"""

    def call(self, x):
        return tf.shape(x)[1]


class ZerosLayer(tf.keras.layers.Layer):
    """Layer for creating zeros tensor with proper shape"""

    def __init__(self, predict_length, **kwargs):
        super(ZerosLayer, self).__init__(**kwargs)
        self.predict_length = predict_length

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.zeros([batch_size, self.predict_length], dtype=tf.float32)
