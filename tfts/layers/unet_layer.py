# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.unet`"""

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Multiply


class ConvbrLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, kernel_size: int, strides: int, dilation: int):
        super(ConvbrLayer, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation

    def build(self, input_shape):
        self.conv1 = Conv1D(
            self.units, kernel_size=self.kernel_size, strides=self.strides, dilation_rate=self.dilation, padding="same"
        )
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        super(ConvbrLayer, self).build(input_shape)

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
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_config(self):
        config = {"units": self.units, "kernel_size": self.kernel_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SeBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Networks"""

    def __init__(self, units):
        super(SeBlock, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.pool = GlobalAveragePooling1D()
        self.fc1 = Dense(self.units // 8, activation="relu")
        self.fc2 = Dense(self.units, activation="sigmoid")
        super(SeBlock, self).build(input_shape)

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
        input = x
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x_out = Multiply()([input, x])
        return x_out

    def get_config(self):
        config = {"units": self.units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReBlock(tf.keras.layers.Layer):
    def __init__(self, units, kernel_size, strides, dilation, use_se):
        super(ReBlock, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.conv_br1 = ConvbrLayer(units, kernel_size, strides, dilation)
        self.conv_br2 = ConvbrLayer(units, kernel_size, strides, dilation)
        if use_se:
            self.se_block = SeBlock(units=units)
        self.use_se = use_se

    def build(self, input_shape):
        super(ReBlock, self).build(input_shape)

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
        x_re = self.conv_br1(x)
        x_re = self.conv_br2(x_re)
        if self.use_se:
            x_re = self.se_block(x_re)
            x_re = Add()([x, x_re])
        return x_re

    def get_config(self):
        config = {"units": self.units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv_br(x, units, kernel_size, strides=1, dilation=1):
    # a function is easier to reuse
    convbr = ConvbrLayer(units=units, kernel_size=kernel_size, strides=strides, dilation=dilation)
    out = convbr(x)
    return out


def se_block(x, units):
    seblock = SeBlock(units)
    out = seblock(x)
    return out


def re_block(x, units, kernel_size, strides=1, dilation=1, use_se=True):
    reblock = ReBlock(units, kernel_size, strides, dilation, use_se=use_se)
    out = reblock(x)
    return out
