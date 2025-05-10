import unittest

from tfts.data.get_data import get_air_passengers, get_data, get_sine


class GetDataTest(unittest.TestCase):
    def test_get_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        n_examples = 100
        train, valid = get_data("sine", train_length, predict_sequence_length, test_size)
        self.assertEqual(train[0].shape, (int(n_examples * (1 - test_size)), train_length, 1))
        self.assertEqual(train[1].shape, (int(n_examples * (1 - test_size)), predict_sequence_length, 1))
        self.assertEqual(valid[0].shape, (int(n_examples * test_size), train_length, 1))
        self.assertEqual(valid[1].shape, (int(n_examples * test_size), predict_sequence_length, 1))

    def test_get_sine_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        n_examples = 200
        train, valid = get_sine(train_length, predict_sequence_length, test_size, n_examples)
        self.assertEqual(train[0].shape, (int(n_examples * (1 - test_size)), train_length, 1))
        self.assertEqual(train[1].shape, (int(n_examples * (1 - test_size)), predict_sequence_length, 1))
        self.assertEqual(valid[0].shape, (int(n_examples * test_size), train_length, 1))
        self.assertEqual(valid[1].shape, (int(n_examples * test_size), predict_sequence_length, 1))

    def test_get_air_passenger_data(self):
        train_length = 10
        predict_sequence_length = 4
        test_size = 0.2
        train, valid = get_air_passengers(train_length, predict_sequence_length, test_size)
        self.assertEqual(train[0].shape[1:], (train_length, 1))
        self.assertEqual(train[1].shape[1:], (predict_sequence_length, 1))
        self.assertEqual(valid[0].shape[1:], (train_length, 1))
        self.assertEqual(valid[1].shape[1:], (predict_sequence_length, 1))
