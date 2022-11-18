"""
python -m unittest -v tests/test_demo.py
"""

import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer as Trainer, build_tfts_model


class DemoTest(unittest.TestCase):
    def test_demo(self):
        train_length = 24
        predict_length = 8

        (x_train, y_train), (x_valid, y_valid) = tfts.get_data("sine", train_length, predict_length, test_size=0.2)
        model = AutoModel("seq2seq", predict_length=predict_length)

        trainer = Trainer(model)
        trainer.train((x_train, y_train), (x_valid, y_valid), n_epochs=3)

        pred = trainer.predict(x_valid)
        trainer.plot(history=x_valid, true=y_valid, pred=pred)

    # def test_auto_model(self):
    #     predict_length = 2
    #     for m in ["seq2seq", "wavenet", "transformer"]:
    #         build_tfts_model(m, predict_length=predict_length)
    #
    #     for m in ["seq2seq", "wavenet", "transformer", "rnn", "tcn", "bert", "informer", "autoformer"]:
    #         model = AutoModel(m, predict_length=predict_length)
    #         y = model(
    #             (tf.random.normal([1, 13, 1]), tf.random.normal([1, 13, 3]), tf.random.normal([1, predict_length, 5]))
    #         )
    #         self.assertEqual(y.shape, (1, predict_length, 1))
    #
    #     for m in [
    #         "seq2seq",
    #         "wavenet",
    #         "transformer",
    #         "rnn",
    #         "tcn",
    #         "bert",
    #         # "tft",
    #         "unet",
    #         "informer",
    #         "autoformer",
    #         "nbeats",
    #         # "gan",
    #     ]:
    #         AutoConfig(m)
    #
    # @unittest.skip
    # def test_train(self):
    #     for m in ["seq2seq", "wavenet", "transformer"]:
    #         train, valid = tfts.get_data("sine", test_size=0.1)
    #         model = AutoModel(m, predict_length=8)
    #         trainer = Trainer(model)
    #         trainer.train(train, valid, n_epochs=3)
    #         y_test = trainer.predict(valid[0])
    #         self.assertEqual(y_test.shape, valid[1].shape)
