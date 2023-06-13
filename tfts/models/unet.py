"""
`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_
"""

from typing import Any, Callable, Dict, Optional, Tuple, Type

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, AveragePooling1D, Concatenate, Conv1D, Input, Lambda, UpSampling1D

from tfts.layers.unet_layer import conv_br, re_block

params = {
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class Unet(object):
    """Unet model"""

    def __init__(
        self,
        predict_sequence_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length

        self.AvgPool1D1 = AveragePooling1D(pool_size=2)
        self.AvgPool1D2 = AveragePooling1D(pool_size=4)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, training=True):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        predict_seq_length : _type_
            _description_
        training : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        pool1 = self.AvgPool1D1(x)
        pool2 = self.AvgPool1D2(x)

        encoder_output = self.encoder([x, pool1, pool2])
        decoder_outputs = self.decoder(encoder_output, predict_seq_length=self.predict_sequence_length)

        if self.params["skip_connect_circle"]:
            x_mean = x[:, -self.predict_sequence_length :, 0:1]
            decoder_outputs = decoder_outputs + x_mean
        if self.params["skip_connect_mean"]:
            x_mean = tf.tile(tf.reduce_mean(x[..., 0:1], axis=1, keepdims=True), [1, self.predict_sequence_length, 1])
            decoder_outputs = decoder_outputs + x_mean
        return decoder_outputs


class Encoder(object):
    def __init__(self):
        pass

    def __call__(self, input_tensor, units=64, kernel_size=2, depth=2):
        """_summary_

        Parameters
        ----------
        input_tensor : _type_
            _description_
        units : int, optional
            _description_, by default 64
        kernel_size : int, optional
            _description_, by default 2
        depth : int, optional
            _description_, by default 2

        Returns
        -------
        _type_
            _description_
        """
        x, pool1, pool2 = input_tensor

        x = conv_br(x, units, kernel_size, 1, 1)  # => batch_size * sequence_length * units
        for i in range(depth):
            x = re_block(x, units, kernel_size, 1, 1)
        out_0 = x  # => batch_size * sequence_length * units

        x = conv_br(x, units * 2, kernel_size, 2, 1)
        for i in range(depth):
            x = re_block(x, units * 2, kernel_size, 1, 1)
        out_1 = x  # => batch_size * sequence/2 * units*2

        x = Concatenate()([x, pool1])
        x = conv_br(x, units * 3, kernel_size, 2, 1)
        for i in range(depth):
            x = re_block(x, units * 3, kernel_size, 1, 1)
        out_2 = x  # => batch_size * sequence/2, units*3

        x = Concatenate()([x, pool2])
        x = conv_br(x, units * 4, kernel_size, 4, 1)
        for i in range(depth):
            x = re_block(x, units * 4, kernel_size, 1, 1)
        return [out_0, out_1, out_2, x]


class Decoder(object):
    def __init__(self):
        pass

    def __call__(self, input_tensor, units=64, kernel_size=2, predict_seq_length=1):
        """_summary_

        Parameters
        ----------
        input_tensor : _type_
            _description_
        units : int, optional
            _description_, by default 64
        kernel_size : int, optional
            _description_, by default 2
        predict_seq_length : int, optional
            _description_, by default 1

        Returns
        -------
        _type_
            _description_
        """
        out_0, out_1, out_2, x = input_tensor
        x = UpSampling1D(4)(x)
        x = Concatenate()([x, out_2])
        x = conv_br(x, units * 3, kernel_size, 1, 1)

        x = UpSampling1D(2)(x)
        x = Concatenate()([x, out_1])
        x = conv_br(x, units * 2, kernel_size, 1, 1)

        x = UpSampling1D(2)(x)
        x = Concatenate()([x, out_0])
        x = conv_br(x, units, kernel_size, 1, 1)

        # regression
        x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
        out = Activation("sigmoid")(x)
        out = Lambda(lambda x: 12 * x)(out)
        # Todo: just a tricky way to change the batch*input_seq*1 -> batch_out_seq*1, need a more general way
        out = AveragePooling1D(strides=4)(out)
        return out
