"""Layer for :py:class:`~tfts.models.wavenet`"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import activations, constraints, initializers, regularizers


class ConvTemp(tf.keras.layers.Layer):
    """Temporal convolutional layer"""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        dilation_rate: int = 1,
        activation: str = "relu",
        causal: bool = True,
        kernel_initializer: str = "glorot_uniform",
        name=None,
    ):
        super(ConvTemp, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.causal = causal
        self.kernel_initializer = kernel_initializer
        self.activation = activation

    def build(self, input_shape: Tuple[int]) -> None:

        self.conv = tf.keras.layers.Conv1D(
            kernel_size=self.kernel_size,
            kernel_initializer=initializers.get(self.kernel_initializer),
            filters=self.filters,
            padding="valid",
            dilation_rate=self.dilation_rate,
            activation=activations.get(self.activation),
        )
        super(ConvTemp, self).build(input_shape)

    def call(self, inputs):
        """forward pass

        Parameters
        ----------
        inputs : inputs
            tensor with batch * sequence * features

        Returns
        -------
        tf.Tensor
            _description_
        """
        if self.causal:
            padding_size = (self.kernel_size - 1) * self.dilation_rate
            # padding: dim 1 is batch, [0,0]; dim 2 is time, [padding_size, 0]; dim 3 is feature [0,0]
            inputs = tf.pad(inputs, [[0, 0], [padding_size, 0], [0, 0]])

        outputs = self.conv(inputs)
        return outputs

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation_rate": self.dilation_rate,
            "casual": self.causal,
        }
        base_config = super(ConvTemp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class ConvAttTemp(tf.keras.layers.Layer):
#     """Temp convolutional attention layer"""
#
#     def __init__(self):
#         super(ConvAttTemp, self).__init__()
#         self.temporal_conv = ConvTemp()
#         self.att = SelfAttention()
#
#     def build(self, input_shape: Tuple[Optional[int], ...]):
#         super(ConvAttTemp, self).build(input_shape)
#
#     def call(self, inputs):
#         """_summary_
#
#         Parameters
#         ----------
#         inputs : _type_
#             _description_
#
#         Returns
#         -------
#         _type_
#             _description_
#         """
#         x = inputs
#         x = self.temporal_conv(x)
#         x = self.att(x)
#         return x
#
#     def get_config(self):
#         config = {
#             "units": self.units,
#         }
#         base_config = super(ConvAttTemp, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
