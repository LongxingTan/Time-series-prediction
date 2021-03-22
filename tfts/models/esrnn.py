# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-03
# paper: https://www.sciencedirect.com/science/article/pii/S0169207019301153?via%3Dihub
# other implementations: https://github.com/damitkwr/ESRNN-GPU
#                        https://github.com/kdgutier/esrnn_torch


import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten


params = {

}


class ESRNN(object):
    def __init__(self, custom_model_params):
        params.update(custom_model_params)

    def __call__(self, inputs_shape, training):
        pass

