# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

from tensorflow.python.keras.layers import Layer, LSTM


class RNNLayer(Layer):
    def __init__(self):
        super(RNNLayer, self).__init__()
