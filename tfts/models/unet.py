"""
`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, AveragePooling1D, Concatenate, Conv1D, Input, Lambda, UpSampling1D

from tfts.layers.unet_layer import conv_br, re_block

from .base import BaseConfig, BaseModel


class UnetConfig(BaseConfig):
    model_type: str = "unet"

    def __init__(
        self,
        units: int = 64,
        kernel_size: int = 2,
        depth: int = 2,
        pool_sizes: Tuple[int, int] = (2, 4),
        upsampling_factors: Tuple[int, int, int] = (4, 2, 2),
    ):
        super(UnetConfig, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.depth = depth
        self.pool_sizes = pool_sizes
        self.upsampling_factors = upsampling_factors


class Unet(BaseModel):
    """Unet model for sequence-to-sequence prediction tasks."""

    def __init__(self, predict_sequence_length: int = 1, config=None):
        super(Unet, self).__init__()
        self.config = config if config else UnetConfig()
        self.predict_sequence_length = predict_sequence_length

        self.avg_pool1 = AveragePooling1D(pool_size=self.config.pool_sizes[0])
        self.avg_pool2 = AveragePooling1D(pool_size=self.config.pool_sizes[1])
        self.encoder = Encoder(units=self.config.units, kernel_size=self.config.kernel_size, depth=self.config.depth)
        self.decoder = Decoder(
            upsampling_factors=self.config.upsampling_factors,
            units=self.config.units,
            kernel_size=self.config.kernel_size,
            predict_seq_length=predict_sequence_length,
        )

    def __call__(
        self,
        x: tf.Tensor,
        training: bool = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
            training: Boolean flag for training mode.

        Returns:
            Tensor: Output predictions.
        """
        pool1 = self.avg_pool1(x)
        pool2 = self.avg_pool2(x)

        encoder_output = self.encoder([x, pool1, pool2])
        decoder_outputs = self.decoder(encoder_output)

        return decoder_outputs


class Encoder(tf.keras.layers.Layer):
    """Encoder component for the Unet model."""

    def __init__(self, units: int = 64, kernel_size: int = 2, depth: int = 2):
        super().__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.depth = depth

    def call(self, inputs: tf.Tensor):
        """Forward pass through the encoder.

        Args:
            inputs: Tuple containing the input tensor and pooled tensors.

        Returns:
            Tuple: Encoder outputs.
        """
        x, pool1, pool2 = inputs

        x = conv_br(x, self.units, self.kernel_size, 1, 1)  # => batch_size * sequence_length * units
        for i in range(self.depth):
            x = re_block(x, self.units, self.kernel_size, 1, 1)
        out_0 = x  # => batch_size * sequence_length * units

        x = conv_br(x, self.units * 2, self.kernel_size, 2, 1)
        for i in range(self.depth):
            x = re_block(x, self.units * 2, self.kernel_size, 1, 1)
        out_1 = x  # => batch_size * sequence/2 * units*2

        x = Concatenate()([x, pool1])
        x = conv_br(x, self.units * 3, self.kernel_size, 2, 1)
        for i in range(self.depth):
            x = re_block(x, self.units * 3, self.kernel_size, 1, 1)
        out_2 = x  # => batch_size * sequence/2, units*3

        x = Concatenate()([x, pool2])
        x = conv_br(x, self.units * 4, self.kernel_size, 4, 1)
        for i in range(self.depth):
            x = re_block(x, self.units * 4, self.kernel_size, 1, 1)
        return [out_0, out_1, out_2, x]


class Decoder(tf.keras.layers.Layer):
    """Decoder component for the Unet model."""

    def __init__(self, upsampling_factors, units: int = 64, kernel_size: int = 2, predict_seq_length: int = 1):
        super().__init__()
        self.upsampling_factors = upsampling_factors
        self.units = units
        self.kernel_size = kernel_size
        self.predict_seq_length = predict_seq_length

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Forward pass through the decoder.

        Args:
            inputs: Tuple containing encoder outputs.

        Returns:
            Tensor: Decoder output.
        """
        out_0, out_1, out_2, x = inputs

        x = UpSampling1D(self.upsampling_factors[0])(x)
        x = Concatenate()([x, out_2])
        x = conv_br(x, self.units * 3, self.kernel_size, 1, 1)

        x = UpSampling1D(self.upsampling_factors[1])(x)
        x = Concatenate()([x, out_1])
        x = conv_br(x, self.units * 2, self.kernel_size, 1, 1)

        x = UpSampling1D(self.upsampling_factors[2])(x)
        x = Concatenate()([x, out_0])
        x = conv_br(x, self.units, self.kernel_size, 1, 1)

        x = Conv1D(1, kernel_size=self.kernel_size, strides=1, padding="same")(x)
        x = Activation("sigmoid")(x)

        x = Lambda(lambda x: 12 * x)(x)
        # Todo: just a tricky way to change the batch*input_seq*1 -> batch_out_seq*1, need a more general way for time
        x = AveragePooling1D(strides=4)(x)

        return x
