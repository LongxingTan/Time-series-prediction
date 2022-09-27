# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

import tensorflow as tf
from tensorflow.keras import initializers, activations, constraints, regularizers


class ConvTemp(tf.keras.layers.Layer):
    """ Temporal convolutional layer
    """
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        dilation_rate=1,
        activation=None,
        causal=True,
        kernel_initializer='glorot_uniform',
        name=None):

        super(ConvTemp, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.causal = causal
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):  # Create the weights
        self.conv = tf.keras.layers.Conv1D(
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            filters=self.filters,
            padding='valid',
            dilation_rate=self.dilation_rate,
            activation=self.activation)
        super(ConvTemp, self).build(input_shape)

    def call(self, input):
        if self.causal:
            padding_size = (self.kernel_size - 1) * self.dilation_rate
            # padding: 1st dim is batch, [0,0]; 2nd dim is time, [padding_size, 0]; 3rd dim is feature [0,0]
            input = tf.pad(input, [[0, 0], [padding_size, 0], [0, 0]])

        output = self.conv(input)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'casual': self.causal,
        }
        base_config = super(ConvTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvAttTemp(tf.keras.layers.Layer):
    """  Temporal convolutional attention layer

    """
    def __init__(self):
        super(ConvAttTemp, self).__init__()
        self.temporal_conv = ConvTemporal()
        self.att = SelfAttention()

    def build(self, input_shape):
        super(TemporalConvAtt, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = self.temporal_conv(x)
        x = self.att(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(ConvAttTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
