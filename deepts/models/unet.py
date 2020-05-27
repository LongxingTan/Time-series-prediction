# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-03
# paper:
# other implementations: https://www.kaggle.com/super13579/u-net-1d-cnn-with-pytorch
#                        https://www.kaggle.com/kmat2019/u-net-1d-cnn-with-keras


import tensorflow as tf
from tensorflow.keras.layers import (Input, AveragePooling1D, Add, UpSampling1D, Concatenate, Lambda)
from deepts.layers.unet_layer import *


params={}


class Unet(object):
    def __init__(self, custom_model_params):
        self.params=params.update(custom_model_params)
        self.AvgPool1D1 = AveragePooling1D(pool_size=2)
        self.AvgPool1D2 = AveragePooling1D(pool_size=4)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, predict_seq_length, training=True):
        pool1=self.AvgPool1D1(x)
        pool2=self.AvgPool1D2(x)

        encoder_output=self.encoder([x,pool1,pool2])
        decoder_output=self.decoder(encoder_output, predict_seq_length=predict_seq_length)
        return decoder_output


class Encoder(object):
    def __init__(self):
        pass

    def __call__(self, input_tensor,units=64,kernel_size=2,depth=2):
        x,pool1,pool2=input_tensor

        x = conv_br(x, units, kernel_size, 1, 1)  # => batch_size * sequence_length * units
        for i in range(depth):
            x = re_block(x, units, kernel_size, 1, 1)
        out_0 = x  # => batch_size * sequence_length * units

        x = conv_br(x, units * 2, kernel_size, 2, 1)
        for i in range(depth):
            x = re_block(x, units * 2, kernel_size, 1,1)
        out_1 = x  # => batch_size * sequence/2 * units*2

        x = Concatenate()([x, pool1])
        x = conv_br(x, units * 3, kernel_size, 2, 1)
        for i in range(depth):
            x = re_block(x, units * 3, kernel_size, 1,1)
        out_2 = x  # => batch_size * sequence/2, units*3

        x = Concatenate()([x, pool2])
        x = conv_br(x, units * 4, kernel_size, 4, 1)
        for i in range(depth):
            x = re_block(x, units * 4, kernel_size, 1,1)
        return [out_0,out_1,out_2, x]


class Decoder(object):
    def __init__(self):
        pass

    def __call__(self, input_tensor, units=64, kernel_size=2, predict_seq_length=1):
        out_0,out_1,out_2,x=input_tensor
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
        out = Lambda(lambda x: 12*x)(out)
        out = AveragePooling1D(strides=4)(out)  # Todo: just a tricky way to change the batch*input_seq*1 -> batch_out_seq*1, need a more general way

        return out

