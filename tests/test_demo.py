"""
python -m unittest -v tests/test_demo.py
"""

import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer as Trainer


class DemoTest(unittest.TestCase):
    def test_demo(self):
        train_length = 24
        predict_length = 8

        (x_train, y_train), (x_valid, y_valid) = tfts.get_data("sine", train_length, predict_length, test_size=0.2)

        config = AutoConfig.for_model("seq2seq")
        model = AutoModel.from_config(config, predict_length=predict_length)

        trainer = Trainer(model)
        trainer.train((x_train, y_train), (x_valid, y_valid), epochs=2)

        pred = trainer.predict(x_valid)
        # trainer.plot(history=x_valid, true=y_valid, pred=pred)
        print(pred.shape)

    def test_demo2(self):
        train_length = 24
        predict_length = 8

        (x_train, y_train), (x_valid, y_valid) = tfts.get_data("sine", train_length, predict_length, test_size=0.2)
        config = AutoConfig.for_model("seq2seq")
        model = AutoModel.from_config(config=config, predict_length=predict_length)
        print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

        trainer = Trainer(model)
        trainer.train((x_train, y_train), epochs=2)

        pred = trainer.predict(x_valid)
        # trainer.plot(history=x_valid, true=y_valid, pred=pred)
        print(pred.shape)

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
    #         trainer.train(train, valid, epochs=3)
    #         y_test = trainer.predict(valid[0])
    #         self.assertEqual(y_test.shape, valid[1].shape)
