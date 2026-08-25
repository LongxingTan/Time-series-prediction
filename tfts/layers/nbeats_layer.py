"""N-BEATS building blocks (TensorFlow), aligned to the reference implementation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer


def _linspace(backcast_length: int, forecast_length: int, centered: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (backcast, forecast) time grids"""
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float64)
    return lin_space[:backcast_length], lin_space[backcast_length:]


def _frequency_grid(num_frequencies: int, backcast_length: int, forecast_length: int, min_period: int):
    return np.linspace(0, (backcast_length + forecast_length) / min_period, num_frequencies)


class TrendBlock(Layer):
    """Trend block: polynomial basis extrapolation."""

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        units: int,
        num_block_layers: int = 4,
        thetas_dim: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = units
        self.num_block_layers = num_block_layers
        self.thetas_dim = thetas_dim
        self.dropout = dropout

        # polynomial times basis (degree 0 .. thetas_dim-1), scaled like the reference
        norm = np.sqrt(forecast_length / thetas_dim)
        b_ls, f_ls = _linspace(backcast_length, forecast_length, centered=True)
        powers = np.arange(thetas_dim)[:, None]
        self._backcast_time = b_ls.astype(np.float32)
        self._forecast_time = f_ls.astype(np.float32)
        backcast_basis = (b_ls[None, :] ** powers) * norm  # (thetas_dim, backcast)
        forecast_basis = (f_ls[None, :] ** powers) * norm  # (thetas_dim, forecast)
        self._backcast_basis = backcast_basis.astype(np.float32)
        self._forecast_basis = forecast_basis.astype(np.float32)

    def build(self, input_shape):
        super().build(input_shape)
        self.layers_in = Dense(self.units, activation="relu")
        self.dropouts = [Dropout(self.dropout) for _ in range(self.num_block_layers - 1)]
        self.hidden = [Dense(self.units) for _ in range(self.num_block_layers - 1)]
        self.theta = Dense(self.thetas_dim, use_bias=False)

    def call(self, inputs, training=None):
        x = inputs
        x = self.layers_in(x)
        for drop, dense in zip(self.dropouts, self.hidden):
            x = tf.nn.relu(dense(drop(x, training=training)))
        theta = self.theta(x)  # (batch, thetas_dim)
        backcast = tf.matmul(theta, self._backcast_basis)  # (batch, backcast)
        forecast = tf.matmul(theta, self._forecast_basis)  # (batch, forecast)
        return backcast, forecast

    def compute_output_shape(self, input_shape):
        return (
            (input_shape[0], self.backcast_length),
            (input_shape[0], self.forecast_length),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backcast_length": self.backcast_length,
                "forecast_length": self.forecast_length,
                "units": self.units,
                "num_block_layers": self.num_block_layers,
                "thetas_dim": self.thetas_dim,
                "dropout": self.dropout,
            }
        )
        return config


class SeasonalityBlock(Layer):
    """Seasonality block: Fourier basis extrapolation."""

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        units: int,
        num_block_layers: int = 4,
        thetas_dim: Optional[int] = None,
        min_period: int = 7,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thetas_dim is None:
            thetas_dim = forecast_length  # reference behaviour when no explicit harmonics
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = units
        self.num_block_layers = num_block_layers
        self.thetas_dim = thetas_dim
        self.min_period = min_period
        self.dropout = dropout

        b_ls, f_ls = _linspace(backcast_length, forecast_length, centered=False)
        p1 = thetas_dim // 2
        p2 = thetas_dim - p1
        freqs1 = _frequency_grid(p1, backcast_length, forecast_length, min_period)
        freqs2 = _frequency_grid(p2, backcast_length, forecast_length, min_period)
        s1_b = np.cos(2.0 * np.pi * freqs1[:, None] * b_ls[None, :])
        s2_b = np.sin(2.0 * np.pi * freqs2[:, None] * b_ls[None, :])
        s1_f = np.cos(2.0 * np.pi * freqs1[:, None] * f_ls[None, :])
        s2_f = np.sin(2.0 * np.pi * freqs2[:, None] * f_ls[None, :])
        self._backcast_basis = np.vstack([s1_b, s2_b]).astype(np.float32)
        self._forecast_basis = np.vstack([s1_f, s2_f]).astype(np.float32)

    def build(self, input_shape):
        super().build(input_shape)
        self.layers_in = Dense(self.units, activation="relu")
        self.dropouts = [Dropout(self.dropout) for _ in range(self.num_block_layers - 1)]
        self.hidden = [Dense(self.units) for _ in range(self.num_block_layers - 1)]
        self.theta = Dense(self.thetas_dim, use_bias=False)

    def call(self, inputs, training=None):
        x = inputs
        x = self.layers_in(x)
        for drop, dense in zip(self.dropouts, self.hidden):
            x = tf.nn.relu(dense(drop(x, training=training)))
        theta = self.theta(x)  # (batch, thetas_dim)
        backcast = tf.matmul(theta, self._backcast_basis)  # (batch, backcast)
        forecast = tf.matmul(theta, self._forecast_basis)  # (batch, forecast)
        return backcast, forecast

    def compute_output_shape(self, input_shape):
        return (
            (input_shape[0], self.backcast_length),
            (input_shape[0], self.forecast_length),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backcast_length": self.backcast_length,
                "forecast_length": self.forecast_length,
                "units": self.units,
                "num_block_layers": self.num_block_layers,
                "thetas_dim": self.thetas_dim,
                "min_period": self.min_period,
                "dropout": self.dropout,
            }
        )
        return config


class GenericBlock(Layer):
    """Generic (fully-connected) N-BEATS block, kept for API compatibility.

    The trend/seasonality interpretable blocks are used for parity; this generic MLP
    block produces a backcast/forecast pair via a wider ``theta`` output split in two.
    """

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        units: int,
        num_block_layers: int = 4,
        thetas_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thetas_dim is None:
            thetas_dim = backcast_length + forecast_length
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = units
        self.num_block_layers = num_block_layers
        self.thetas_dim = thetas_dim
        self.dropout = dropout

    def build(self, input_shape):
        super().build(input_shape)
        self.layers_in = Dense(self.units, activation="relu")
        self.dropouts = [Dropout(self.dropout) for _ in range(self.num_block_layers - 1)]
        self.hidden = [Dense(self.units) for _ in range(self.num_block_layers - 1)]
        self.theta = Dense(self.thetas_dim, use_bias=False)

    def call(self, inputs, training=None):
        x = inputs
        x = self.layers_in(x)
        for drop, dense in zip(self.dropouts, self.hidden):
            x = tf.nn.relu(dense(drop(x, training=training)))
        theta = self.theta(x)
        backcast = theta[:, : self.backcast_length]
        forecast = theta[:, -self.forecast_length :]
        return backcast, forecast

    def compute_output_shape(self, input_shape):
        return (
            (input_shape[0], self.backcast_length),
            (input_shape[0], self.forecast_length),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backcast_length": self.backcast_length,
                "forecast_length": self.forecast_length,
                "units": self.units,
                "num_block_layers": self.num_block_layers,
                "thetas_dim": self.thetas_dim,
                "dropout": self.dropout,
            }
        )
        return config
