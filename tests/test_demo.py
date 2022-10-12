"""
python -m unittest -v tests/test_demo.py
"""

import unittest

import tfts
from tfts import AutoModel, KerasTrainer as Trainer


class DemoTest(unittest.TestCase):
    def test_demo(self):
        train, valid = tfts.get_data("sine")
        print(train[0].shape, train[1].shape)

        model = AutoModel("seq2seq", predict_length=8)

        trainer = Trainer(model)
        trainer.train(train, valid, n_epochs=3)
        trainer.predict(valid[0])

    @unittest.skip
    def test_train(self):
        for m in ["seq2seq", "wavenet", "transformer"]:
            train, valid = tfts.load_data("sine", test_size=0.1)
            model = AutoModel(m, predict_length=8)
            trainer = Trainer(model)
            trainer.train(train, valid, n_epochs=3)
            y_test = trainer.predict(valid[0])
            self.assertEqual(y_test.shape, valid[1].shape)
