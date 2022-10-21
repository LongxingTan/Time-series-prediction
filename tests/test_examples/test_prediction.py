import unittest

import tensorflow as tf

from examples.run_prediction import parse_args, run_train
from examples.utils import set_seed


class PredictionTest(unittest.TestCase):
    def test_train(self):
        class args(object):
            seed = 315
            use_data = "sine"
            use_model = "rnn"
            train_length = 10
            predict_length = 5
            n_epochs = 3
            batch_size = 16
            learning_rate = 0.003

        set_seed(args.seed)
        run_train(args)
