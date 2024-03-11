#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
"""AutoConfig to set up models custom config"""

import collections

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


class BaseConfig(object):
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat
