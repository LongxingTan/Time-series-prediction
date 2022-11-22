#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""AutoModel to choose different models"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf

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

    def __init__(
        self,
        use_model: str,
        predict_length: int = 1,
        custom_model_params: Optional[Dict[str, Any]] = None,
        custom_model_head: Optional[Callable] = None,
    ):
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
        # elif use_model.lower() == "tft":
        #    self.model = TFTransformer(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "unet":
            self.model = Unet(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        elif use_model.lower() == "nbeats":
            self.model = NBeats(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        # elif use_model.lower() == "gan":
        #     self.model = GAN(predict_sequence_length=predict_length, custom_model_params=custom_model_params)
        else:
            raise ValueError("unsupported model of {} yet".format(use_model))

    def __call__(
        self, x: Union[tf.data.Dataset, Tuple[np.array], Tuple[pd.DataFrame], List[np.array], List[pd.DataFrame]]
    ):
        """automodel callable

        Parameters
        ----------
        x : tf.data.Dataset, np.array
            model inputs

        Returns
        -------
        tf.Tensor
            model output
        """
        # if isinstance(x, (list, tuple)):
        #     assert len(x[0].shape) == 3, "The expected inputs dimension is 3, while get {}".format(len(x[0].shape))
        return self.model(x)

    def build_model(self, inputs):
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
