#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""AutoModel to choose different models"""

import tensorflow as tf
from tensorflow.keras.layers import Input

from reference.gan import GAN
from reference.temporal_fusion_transformer import TFTransformer
from tfts.models.autoformer import AutoFormer
from tfts.models.bert import Bert
from tfts.models.informer import Informer
from tfts.models.nbeats import NBeats
from tfts.models.rnn import RNN
from tfts.models.seq2seq import Seq2seq
from tfts.models.tcn import TCN
from tfts.models.transformer import Transformer
from tfts.models.unet import Unet
from tfts.models.wavenet import WaveNet


class AutoModel(object):
    """AutoModel"""

    def __init__(self, use_model, predict_length=1, custom_model_params=None, custom_model_head=None):
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
        elif use_model.lower() == "informer":
            self.model = Informer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "autoformer":
            self.model = AutoFormer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "tft":
            self.model = TFTransformer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "unet":
            self.model = Unet(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "nbeats":
            self.model = NBeats(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "gan":
            self.model = GAN(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        else:
            raise ValueError("unsupported model of {} yet".format(use_model))

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(x, (list, tuple)):
            assert len(x[0].shape) == 3, "The expected inputs dimension is 3, while get {}".format(len(x[0].shape))
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
