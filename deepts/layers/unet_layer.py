
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-03


import tensorflow as tf
from tensorflow.keras.layers import (Conv1D,BatchNormalization,Activation,Dense,GlobalAveragePooling1D,add)


# Todo: unfinished yet
class ConbrLayer(tf.keras.layers.Layer):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(ConbrLayer, self).__init__()
        pass

    def build(self, input_shape):
        self.conv1 = Conv1D()
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        super(ConbrLayer, self).build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeBlock(tf.keras.layers.Layer):
    def __init__(self, in_layer, out_layer):
        super(SeBlock, self).__init__()

    def build(self, input_shape):
        self.pool = GlobalAveragePooling1D()
        self.conv1 = Conv1D()
        self.conv2 = Conv1D()
        self.fc1 = Dense()
        self.fc2 = Dense()
        self.relu = Activation('relu')
        self.sogmoid = Activation('sigmoid')
        super(SeBlock, self).build(input_shape)

    def call(self, x):
        x_se = self.pool(x)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        x_out = add(x, x_se)
        return x_out


class ReBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ReBlock, self).__init__()

    def build(self, input_shape):
        self.cbr1 = ConbrLayer()
        self.cbr2 = ConbrLayer()
        self.se_block = SeBlock()
        super(ReBlock, self).build(input_shape)

    def call(self, x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.se_block(x_re)
        x_out = add(x, x_re)
        return x_out
