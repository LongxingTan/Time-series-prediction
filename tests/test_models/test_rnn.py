
import unittest
import tensorflow as tf
from tfts.models.rnn import RNN


class RNNTest(unittest.TestCase):
    def test_model(self):
        custom_model_params = {}
        model = RNN(custom_model_params)

        x = tf.random.normal([2, ])
        y = model(x)
        self.assertEqual(y.shape, (2, ), 'incorrect output shape')

    def test_train(self):
        train_data, valid_data = tfts.load_data('passenger', split=0.2)
        model = RNN(custom_model_params)
        trainer = Trainer(model)
        trainer.train(train_data, valid_data)
        valid_pred = trainer.predict(valid_data)
        self.assertEqual(valid_pred.shape, ())


if __name__ == '__main__':
    unittest.main()
