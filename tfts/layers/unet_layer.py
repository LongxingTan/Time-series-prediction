"""Layer for :py:class:`~tfts.models.unet`"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Multiply


class ConvbrLayer(tf.keras.layers.Layer):
    """
    1D Convolution + BatchNorm + ReLU block.
    """

    def __init__(self, units: int, kernel_size: int, strides: int, dilation: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.conv1 = Conv1D(
            self.units, kernel_size=self.kernel_size, strides=self.strides, dilation_rate=self.dilation, padding="same"
        )
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for ConvbrLayer.
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        Returns:
            Output tensor after conv, batchnorm, relu.
        """
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = {
            "units": self.units,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation": self.dilation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SeBlock(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.pool = GlobalAveragePooling1D()
        self.fc1 = Dense(self.units // 8, activation="relu")
        self.fc2 = Dense(self.units, activation="sigmoid")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for SeBlock.
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
        Returns:
            Output tensor with channel-wise recalibration.
        """
        input_tensor = x
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.expand_dims(x, axis=1)  # (batch, 1, channels)
        x_out = Multiply()([input_tensor, x])
        return x_out

    def get_config(self) -> Dict[str, Any]:
        config = {"units": self.units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReBlock(tf.keras.layers.Layer):
    """
    Residual block with two Convbr layers and optional SE block.
    """

    def __init__(self, units: int, kernel_size: int, strides: int, dilation: int, use_se: bool, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.use_se = use_se

    def build(self, input_shape: Tuple[Optional[int], ...]):
        self.conv_br1 = ConvbrLayer(self.units, self.kernel_size, self.strides, self.dilation)
        self.conv_br2 = ConvbrLayer(self.units, self.kernel_size, self.strides, self.dilation)
        if self.use_se:
            self.se_block = SeBlock(units=self.units)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for ReBlock.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after two Convbr layers, optional SE, and residual add.
        """
        x_re = self.conv_br1(x)
        x_re = self.conv_br2(x_re)
        if self.use_se:
            x_re = self.se_block(x_re)
            x_re = Add()([x, x_re])
        return x_re

    def get_config(self) -> Dict[str, Any]:
        config = {
            "units": self.units,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation": self.dilation,
            "use_se": self.use_se,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
