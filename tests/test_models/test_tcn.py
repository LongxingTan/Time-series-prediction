import functools
import unittest

import tensorflow as tf

import tfts
from tfts import AutoModel, Trainer
from tfts.models.tcn import TCN


class TCNTest(unittest.TestCase):
    def test_model(self):
        custom_model_params = {}
        model = TCN(custom_model_params)

        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16,), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.load_data("sine", test_size=0.1)
        backbone = AutoModel("tcn", predict_sequence_length=8)
        model = functools.partial(backbone.build_model, input_shape=[24, 2])
        trainer = Trainer(model)
        trainer.train(train, valid)
        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)


if __name__ == "__main__":
    unittest.main()
