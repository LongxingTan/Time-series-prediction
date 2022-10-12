#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

import tensorflow as tf
from tensorflow.keras.layers import Input

from tfts.models import RNN, TCN, Bert, NBeats, Seq2seq, Transformer, Unet, WaveNet


class AutoModel(object):
    def __init__(self, use_model, predict_length, custom_model_params=None, custom_model_head=None):
        if use_model.lower() == "seq2seq":
            self.model = Seq2seq(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "rnn":
            self.model = RNN(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "wavenet":
            self.model = WaveNet(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "tcn":
            self.model = TCN(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "transformer":
            self.model = Transformer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "bert":
            self.model = Bert(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        else:
            raise ValueError("unsupported model of {} yet".format(use_model))

    def __call__(self, x):
        return self.model(x)

    def build_model(self, input_shape):
        inputs = Input(input_shape)
        outputs = self.model(inputs)
        return tf.keras.Model(inputs, outputs)


def build_tfts_model(use_model, predict_length, custom_model_params=None):
    if use_model.lower() == "seq2seq":
        Model = Seq2seq(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
    elif use_model.lower() == "wavenet":
        Model = WaveNet(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
    elif use_model.lower() == "transformer":
        Model = Transformer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
    else:
        raise ValueError("unsupported use_model of {} yet".format(use_model))
    return Model
