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

from .base import BaseConfig, BaseModel


class AutoModel(object):
    """tftf auto model"""

    def __init__(
        self,
        use_model: str,
        predict_length: int = 1,
        custom_model_params: Optional[Dict[str, object]] = None,
        custom_model_head: Optional[Callable] = None,
        include_top: bool = False,
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
        self,
        x: Union[tf.data.Dataset, Tuple[np.ndarray], Tuple[pd.DataFrame], List[np.ndarray], List[pd.DataFrame]],
        return_dict: Optional[bool] = None,
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
        return self.model(x, return_dict=return_dict)

    def build_model(self, inputs):
        outputs = self.model(inputs)
        return tf.keras.Model([inputs], [outputs])  # to handles the Keras symbolic tensors for tf2.3.1

    @classmethod
    def from_config(cls, name: str):
        return

    @classmethod
    def from_pretrained(cls, name: str):
        return


class AutoModelForPrediction(BaseModel):
    """tfts model for prediction"""

    def __init__(self, use_model):
        super(AutoModelForPrediction, self).__init__()
        self.model = AutoModel(use_model=use_model)

    def __call__(self):
        return


class AutoModelForClassification(BaseModel):
    """tfts model for classification"""

    def __init__(self):
        super(AutoModelForClassification, self).__init__()
        pass

    def __call__(
        self,
    ):
        return


class AutoModelForAnomaly(BaseModel):
    """tfts model for anomaly detection"""

    def __init__(self):
        super(AutoModelForAnomaly, self).__init__()
        pass

    def __call__(self, *args, **kwargs):
        return


class AutoModelForSegmentation(BaseModel):
    """tfts model for time series segmentation"""

    def __init__(self):
        super(AutoModelForSegmentation, self).__init__()
        pass

    def __call__(self, *args, **kwargs):
        return
