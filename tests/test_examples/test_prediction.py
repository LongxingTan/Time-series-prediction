import unittest
from unittest.mock import patch

import tensorflow as tf

from examples.run_prediction import parse_args, run_train, set_seed


class PredictionTest(unittest.TestCase):
    def test_parse_args(self):
        with patch("sys.argv", ["parse_args", "--seed", "315"]):
            args = parse_args()
            self.assertEqual(args.seed, 315)

    def test_train(self):
        class args(object):
            seed = 315
            use_data = "sine"
            use_model = "rnn"
            train_length = 10
            predict_length = 5
            epochs = 2
            batch_size = 32
            learning_rate = 0.003

        set_seed(args.seed)
        run_train(args)
