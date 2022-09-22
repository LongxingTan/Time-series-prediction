"""
python -m unittest -v tests/test_demo.py
"""

import unittest
import functools
import tensorflow as tf
from tensorflow.keras.layers import Input
import tfts
from tfts import KerasTrainer as Trainer
from tfts import AutoModel


def build_model(model):
    inputs = Input([24, 2])
    outputs = model(inputs)
    return tf.keras.Model(inputs, outputs)


class DemoTest(unittest.TestCase):
    def test_demo(self):
        train, valid = tfts.load('sine', split=0.1)
        print(train[0].shape, train[1].shape)

        backbone = AutoModel('transformer', predict_sequence_length=8)
        model = functools.partial(build_model, backbone)
        trainer = Trainer(model)
        trainer.train(train, valid)
        # trainer.predict(x_valid)

