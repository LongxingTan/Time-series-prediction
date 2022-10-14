# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.nbeats`"""

import math

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense


class NBeatsLayer(tf.keras.layers.Layer):
    def __init__(self, units, thetas_dim, share_thetas):
        super(NBeatsLayer, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.share_thetas = share_thetas

        self.fc1 = Dense(units=self.units, activation="relu")
        self.fc2 = Dense(units=self.units, activation="relu")
        self.fc3 = Dense(units=self.units, activation="relu")
        self.fc4 = Dense(units=self.units, activation="relu")

        if self.share_thetas:
            self.theta_b_fn = self.theta_f_fn = Dense(units=self.thetas_dim, activation="relu", use_bias=False)
        self.theta_b_fn = Dense(units=self.thetas_dim, activation="relu", use_bias=False)
        self.theta_f_fn = Dense(units=self.thetas_dim, activation="relu", use_bias=False)

    def build(self, input_shape):
        super(NBeatsLayer, self).build(input_shape)

    def call(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def linspace(self, backcast_length, forecast_length):
        lin_space = tf.linspace(-float(backcast_length), float(forecast_length), backcast_length + forecast_length)
        b_ls = lin_space[:backcast_length]
        f_ls = lin_space[backcast_length:]
        return b_ls, f_ls

    def get_config(self):
        return


class GenericBlock(NBeatsLayer):
    def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_thetas=False):
        super(GenericBlock, self).__init__(units, thetas_dim, share_thetas)
        self.backcast_fn = Dense(units=backcast_length)
        self.forecast_fn = Dense(units=forecast_length)

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = super(GenericBlock, self).call(x)
        theta_b = self.theta_b_fn(x)
        theta_f = self.theta_f_fn(x)
        backcast = self.backcast_fn(theta_b)
        forecast = self.forecast_fn(theta_f)
        return backcast, forecast


class TrendBlock(NBeatsLayer):
    def __init__(self, units, thetas_dim, backcast_length, forecast_length):
        super(TrendBlock, self).__init__(units, thetas_dim, share_thetas=True)
        self.backcast_linspace, self.forecast_linspace = self.linspace(backcast_length, forecast_length)

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = super(TrendBlock, self).call(x)
        theta_b = self.theta_b_fn(x)
        theta_f = self.theta_f_fn(x)
        backcast = self.trend_model(theta_b, self.backcast_linspace)
        forecast = self.trend_model(theta_f, self.forecast_linspace)
        return backcast, forecast

    def trend_model(self, thetas, t):
        p = thetas.get_shape().as_list()[-1]
        t = tf.transpose(tf.stack([tf.math.pow(t, i) for i in range(p)], axis=0))
        t = tf.cast(t, tf.float32)
        return tf.linalg.matmul(thetas, t, transpose_b=True)


class SeasonalityBlock(NBeatsLayer):
    def __init__(self, units, thetas_dim, backcast_length, forecast_length):
        super(SeasonalityBlock, self).__init__(units, thetas_dim, share_thetas=True)
        self.backcast_linspace, self.forecast_linspace = self.linspace(backcast_length, forecast_length)

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = super(SeasonalityBlock, self).call(x)
        theta_b = self.theta_b_fn(x)
        theta_f = self.theta_f_fn(x)
        backcast = self.seasonality_model(theta_b, self.backcast_linspace)
        forecst = self.seasonality_model(theta_f, self.forecast_linspace)
        return backcast, forecst

    def seasonality_model(self, thetas, t):
        p = thetas.get_shape().as_list()[-1]
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = tf.stack([tf.math.cos(2 * math.pi * i * t) for i in range(p1)], axis=0)
        s1 = tf.cast(s1, tf.float32)
        s2 = tf.stack([tf.math.sin(2 * math.pi * i * t) for i in range(p2)], axis=0)
        s2 = tf.cast(s2, tf.float32)
        s = tf.concat([s1, s2], axis=0)
        return tf.linalg.matmul(thetas, s, transpose_b=True)
