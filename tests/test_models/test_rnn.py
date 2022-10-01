import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, Trainer
from tfts.models.rnn import RNN


class RNNTest(unittest.TestCase):
    def test_model(self):
        custom_model_params = {}
        model = RNN(custom_model_params)

        x = tf.random.normal(
            [
                2,
            ]
        )
        y = model(x)
        self.assertEqual(y.shape, (2,), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.load_data("sine", test_size=0.1)
        backbone = AutoModel("rnn", predict_sequence_length=8)
        model = functools.partial(backbone.build_model, input_shape=[24, 2])
        trainer = Trainer(model)
        trainer.train(train, valid)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)


if __name__ == "__main__":
    unittest.main()
