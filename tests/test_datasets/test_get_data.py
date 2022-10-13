import unittest

from tfts.datasets.get_data import get_data


class GetDataTest(unittest.TestCase):
    def test_get_sine_data(self):
        train_length = 10
        predict_length = 4
        test_size = 0.2
        n_examples = 100
        train, valid = get_data("sine", train_length, predict_length, test_size)
        self.assertEqual(train[0].shape, (int(n_examples * (1 - test_size)), train_length, 2))
        self.assertEqual(train[1].shape, (int(n_examples * (1 - test_size)), predict_length, 1))
        self.assertEqual(valid[0].shape, (int(n_examples * test_size), train_length, 2))
        self.assertEqual(valid[1].shape, (int(n_examples * test_size), predict_length, 1))

    def test_get_air_passenger_data(self):
        get_data("airpassengers")
