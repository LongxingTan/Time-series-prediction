# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf
from tensorflow.python.keras import initializers, activations, constraints, regularizers


class Dense3D(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 bias_initializer="zeros",
                 trainable=True,
                 name=None):
        super(Dense3D, self).__init__(trainable=trainable, name=name)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        inputs_units = int(input_shape[-1])  # input.get_shape().as_list()[-1]
        self.kernel = self.add_weight('kernel',
                                      shape=[inputs_units, self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=tf.float32,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight("bias",
                                        shape=[self.units],
                                        initializer=self.bias_initializer,
                                        dtype=self.dtype,
                                        trainable=True)
        super(Dense3D, self).build(input_shape)

    def call(self, inputs):
        score = tf.einsum('ijk,kl->ijl', inputs, self.kernel)

        if self.use_bias:
            score += self.bias

        if self.activation is not None:
            score = self.activation(score)
        return score


class ConvTime(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 causal=True,
                 kernel_initializer='glorot_uniform',
                 trainable=True,
                 name=None):
        super(ConvTime, self).__init__(trainable=trainable, name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.causal = causal
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):  # Create the weights
        self.conv = tf.keras.layers.Conv1D(kernel_size=self.kernel_size,
                                           kernel_initializer=self.kernel_initializer,
                                           filters=self.filters,
                                           padding='VALID',
                                           dilation_rate=self.dilation_rate,
                                           activation=tf.nn.relu)
        super(ConvTime, self).build(input_shape)

    def call(self, input):
        if self.causal:
            padding_size = (self.kernel_size - 1) * self.dilation_rate
            input = tf.pad(input, [[0, 0], [padding_size, 0], [0, 0]])

        score = self.conv(input)
        return score
