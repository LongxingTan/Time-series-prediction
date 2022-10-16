"""Test the TFT model"""

import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.tft import TemporalFusionTransformer
