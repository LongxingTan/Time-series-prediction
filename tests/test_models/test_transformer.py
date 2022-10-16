import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.transformer import Transformer


class TransformerTest(unittest.TestCase):
    def test_encoder(self):
        pass

    def test_decoder(self):
        pass

    def test_decoder2(self):
        pass

    def test_model(self):
        predict_sequence_length = 8
        custom_model_params = {}
        model = Transformer(predict_sequence_length, custom_model_params)
        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        model = AutoModel("rnn", predict_length=8)
        trainer = KerasTrainer(model)
        trainer.train(train, valid, n_epochs=3)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)
