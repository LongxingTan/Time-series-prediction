# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.nbeats`"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense


class GenericBlock(tf.keras.layers.Layer):
    """Generic block"""

    def __init__(
        self, train_sequence_length: int, predict_sequence_length: int, hidden_size: int, n_block_layers: int = 4
    ):
        super(GenericBlock, self).__init__()
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_block_layers = n_block_layers

    def build(self, input_shape):
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(self.train_sequence_length + self.predict_sequence_length, use_bias=False, activation=None)
        super(GenericBlock, self).build(input_shape)

    def call(self, inputs):
        """_summary_

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)
        return x[:, : self.train_sequence_length], x[:, -self.predict_sequence_length :]


class TrendBlock(tf.keras.layers.Layer):
    """Trend block"""

    def __init__(
        self,
        train_sequence_length: int,
        predict_sequence_length: int,
        hidden_size: int,
        n_block_layers: int = 4,
        polynomial_term=2,
    ):
        super().__init__()
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.hidden_size = hidden_size
        self.n_bloack_layers = n_block_layers
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
                tf.math.pow(tf.range(train_sequence_length, dtype=tf.float32) / train_sequence_length, i)[None, :]
                for i in range(self.polynomial_size)
            ],
            axis=0,
        )

    def build(self, input_shape):
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_bloack_layers)]
        self.theta = Dense(2 * self.polynomial_size, use_bias=False, activation=None)

    def call(self, inputs):
        """_summary_

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)
        backcast = tf.einsum("bp,pt->bt", x[:, self.polynomial_size :], self.backcast_time)
        forecast = tf.einsum("bp,pt->bt", x[:, : self.polynomial_size], self.forecast_time)
        return backcast, forecast


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
            * (tf.range(self.train_sequence_length, dtype=tf.float32)[:, None] / train_sequence_length)
            * self.frequency
        )
        self.forecast_grid = (
            2
            * np.pi
            * (tf.range(predict_sequence_length, dtype=tf.float32)[:, None] / predict_sequence_length)
            * self.frequency
        )
        self.backcast_cos_template = tf.transpose(tf.cos(self.backcast_grid))
        self.backcast_sin_template = tf.transpose(tf.sin(self.backcast_grid))
        self.forecast_cos_template = tf.transpose(tf.cos(self.forecast_grid))
        self.forecast_sin_template = tf.transpose(tf.sin(self.forecast_grid))

    def build(self, input_shape):
        self.layers = [Dense(self.hidden_size, activation="relu") for _ in range(self.n_block_layers)]
        self.theta = Dense(self.theta_size, use_bias=False, activation=None)

    def call(self, inputs):
        """_summary_

        Parameters
        ----------
        inputs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.theta(x)

        params_per_harmonic = self.theta_size // 4

        backcast_harmonics_cos = tf.einsum(
            "bp,pt->bt", inputs[:, 2 * params_per_harmonic : 3 * params_per_harmonic], self.backcast_cos_template
        )
        backcast_harmonics_sin = tf.einsum("bp,pt->bt", x[:, 3 * params_per_harmonic :], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = tf.einsum("bp,pt->bt", x[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = tf.einsum(
            "bp,pt->bt", x[:, params_per_harmonic : 2 * params_per_harmonic], self.forecast_sin_template
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
