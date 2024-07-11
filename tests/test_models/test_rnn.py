import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.models.rnn import RNN, RNN2


class RNNTest(unittest.TestCase):
    def test_model(self):
        predict_sequence_length = 8
        model = RNN(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_length=8)
        trainer = KerasTrainer(model)
        trainer.train(train, valid, n_epochs=2)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)

    def test_model2(self):
        predict_sequence_length = 8
        model = RNN2(predict_sequence_length=predict_sequence_length)

        x = tf.random.normal([2, 16, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")


if __name__ == "__main__":
    unittest.main()
