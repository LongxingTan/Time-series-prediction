import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, KerasTrainer, Trainer
from tfts.models.tcn import TCN


class TCNTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = TCN(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        model = AutoModel("tcn", predict_length=8)
        trainer = KerasTrainer(model)
        trainer.train(train, valid, n_epochs=3)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)


if __name__ == "__main__":
    unittest.main()
