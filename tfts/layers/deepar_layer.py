# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.deepar`"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, units: int):
        self.units = units
        super(GaussianLayer, self).__init__()

    def build(self, input_shape: Tuple[Optional[int], ...]):
        in_channels = input_shape[2]
        self.weight1 = self.add_weight(
            name="gauss_w1", shape=(in_channels, self.units), initializer=tf.keras.initializers.GlorotNormal()
        )
        self.weight2 = self.add_weight(
            name="gauss_w2", shape=(in_channels, self.units), initializer=tf.keras.initializers.GlorotNormal()
        )
        self.bias1 = self.add_weight(name="gauss_b1", shape=(self.units,), initializer=tf.keras.initializers.Zeros())
        self.bias2 = self.add_weight(name="gauss_b2", shape=(self.units,), initializer=tf.keras.initializers.Zeros())
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        """Returns mean and standard deviation tensors.

        Args:
          x (tf.Tensor): Input tensor.

        Returns:
          Tuple[tf.Tensor, tf.Tensor]: Mean and standard deviation tensors.
        """
        mu = tf.matmul(x, self.weight1) + self.bias1
        sig = tf.matmul(x, self.weight2) + self.bias2
        sig_pos = tf.math.log1p(tf.math.exp(sig)) + 1e-7
        return mu, sig_pos

    def get_config(self):
        """Returns the configuration of the layer."""
        config = {"units": self.units}
        base_config = super(GaussianLayer, self).get_config()
        return {**base_config, **config}
