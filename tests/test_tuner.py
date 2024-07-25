import unittest

import tfts
from tfts import AutoConfig, AutoModel, AutoTuner, KerasTrainer


class TunerTest(unittest.TestCase):
    def test_tuner(self):
        train_length = 24
        predict_length = 8

        train, valid = tfts.get_data("sine", train_length, predict_length, test_size=0.2)
        print(train[0].shape, valid[0].shape)

        config = AutoConfig.for_model("rnn")
        print(config)
