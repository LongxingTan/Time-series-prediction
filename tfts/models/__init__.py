#! /usr/bin/env python
# encoding=utf-8
# TFTS supported model

from tfts.models.rnn import RNN
from tfts.models.seq2seq import Seq2seq
from tfts.models.tcn import TCN
from tfts.models.wavenet import WaveNet
from tfts.models.bert import Bert
from tfts.models.transformer import Transformer
from tfts.models.informer import Informer
from tfts.models.autoformer import AutoFormer
from tfts.models.unet import Unet
from tfts.models.nbeats import NBeatsNet
from tfts.models.gan import GAN


__all__ = [
    "RNN",
    "Seq2seq",
    "TCN",
    "WaveNet",
    "Bert",
    "Transformer",
    "Informer",
    "AutoFormer"
]
