"""
python -m unittest -v tests/test_demo.py
"""

import unittest

import tfts
from tfts import AutoModel, KerasTrainer as Trainer, build_tfts_model


class DemoTest(unittest.TestCase):
    def test_demo(self):
        train_length = 24
        predict_length = 8

        train, valid = tfts.get_data("sine", train_length, predict_length, test_size=0.2)
        model = AutoModel("seq2seq", predict_length=predict_length)

        trainer = Trainer(model)
        trainer.train(train, valid, n_epochs=3)

        pred = trainer.predict(valid[0])
        trainer.plot(history=valid[0], true=valid[1], pred=pred)

    def test_auto_model(self):
        for m in ["seq2seq", "wavenet", "transformer"]:
            build_tfts_model(m, predict_length=2)

        for m in [
            "seq2seq",
            "wavenet",
            "transformer",
            "rnn",
            "tcn",
            "bert",
            "tft",
            "unet",
            "informer",
            "autoformer",
            "nbeats",
            "gan",
        ]:
            AutoModel(m)

    @unittest.skip
    def test_train(self):
        for m in ["seq2seq", "wavenet", "transformer"]:
            train, valid = tfts.get_data("sine", test_size=0.1)
            model = AutoModel(m, predict_length=8)
            trainer = Trainer(model)
            trainer.train(train, valid, n_epochs=3)
            y_test = trainer.predict(valid[0])
            self.assertEqual(y_test.shape, valid[1].shape)
