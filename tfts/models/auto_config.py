#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""AutoConfig to set up models custom config"""

import json

from tfts.models.autoformer import params as autoformer_params
from tfts.models.bert import params as bert_params
from tfts.models.informer import params as informer_params
from tfts.models.nbeats import params as nbeats_params
from tfts.models.rnn import params as rnn_params
from tfts.models.seq2seq import params as seq2seq_params
from tfts.models.tcn import params as tcn_params
from tfts.models.transformer import params as transformer_params
from tfts.models.unet import params as unet_params
from tfts.models.wavenet import params as wavenet_params


class BaseConfig:
    def __init__(self, **kwargs):
        self.update(kwargs)

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dict__}

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        with open(pretrained_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class AutoConfig:
    """AutoConfig for model"""

    def __init__(self, use_model: str) -> None:
        if use_model.lower() == "seq2seq":
            self.params = seq2seq_params
        elif use_model.lower() == "rnn":
            self.params = rnn_params
        elif use_model.lower() == "wavenet":
            self.params = wavenet_params
        elif use_model.lower() == "tcn":
            self.params = tcn_params
        elif use_model.lower() == "transformer":
            self.params = transformer_params
        elif use_model.lower() == "bert":
            self.params = bert_params
        elif use_model.lower() == "informer":
            self.params = informer_params
        elif use_model.lower() == "autoformer":
            self.params = autoformer_params
        # elif use_model.lower() == "tft":
        #     self.params = tf_transformer_params
        elif use_model.lower() == "unet":
            self.params = unet_params
        elif use_model.lower() == "nbeats":
            self.params = nbeats_params
        # elif use_model.lower() == "gan":
        #     self.params = gan_params
        else:
            raise ValueError("Unsupported model of {} yet".format(use_model))

    def get_config(self):
        return self.params

    def print_config(self) -> None:
        print(self.params)

    def save_config(self):
        return
