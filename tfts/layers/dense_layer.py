# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""Layer for :py:class:`~tfts.models.wavenet` :py:class:`~tfts.models.transformer`"""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout


class DenseTemp(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        activation: Optional[str] = None,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        use_bias: bool = True,
        bias_initializer="zeros",
        trainable=True,
        name=None,
    ):
        super(DenseTemp, self).__init__(trainable=trainable, name=name)
        self.hidden_size = hidden_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        inputs_units = int(input_shape[-1])  # input.get_shape().as_list()[-1]
        self.kernel = self.add_weight(
            "kernel",
            shape=[inputs_units, self.hidden_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=tf.float32,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias", shape=[self.hidden_size], initializer=self.bias_initializer, dtype=self.dtype, trainable=True
            )
        super(DenseTemp, self).build(input_shape)

    def call(self, inputs):
        """Computes the output of the layer.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            output: Tensor of shape (batch_size, sequence_length, hidden_size)
        """
        output = tf.einsum("ijk,kl->ijl", inputs, self.kernel)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
        }
        base_config = super(DenseTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, filter_size: int, relu_dropout: float = 0.0):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

    def build(self, input_shape):
        self.filter_dense_layer = Dense(self.filter_size, use_bias=True, activation="relu")
        self.output_dense_layer = Dense(self.hidden_size, use_bias=True)
        self.drop = Dropout(self.relu_dropout)
        super(FeedForwardNetwork, self).build(input_shape)

    def call(self, x):
        """Feed Forward Network

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        output = self.filter_dense_layer(x)
        output = self.drop(output)
        output = self.output_dense_layer(output)
        return output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }
        base_config = super(FeedForwardNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
