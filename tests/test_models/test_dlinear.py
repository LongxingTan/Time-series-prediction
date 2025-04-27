from typing import Any, Dict
import unittest

import numpy as np
import tensorflow as tf

import tfts
from tfts import AutoConfig, AutoModel, KerasTrainer, Trainer
from tfts.models.dlinear import DLinear, DLinearConfig


class DLinearTest(unittest.TestCase):
    def test_model(self):
        train_sequence_length = 14
        predict_sequence_length = 7
        config = AutoConfig.for_model("dlinear")
        config.channels = 3  # number of features
        model = DLinear(predict_sequence_length=predict_sequence_length, config=config)

        x = tf.random.normal([2, train_sequence_length, 3])
        y = model(x)
        self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")

    def test_train(self):
        train, valid = tfts.get_data("sine", test_size=0.1)  # Change this to the appropriate dataset
        config = AutoConfig.for_model("dlinear")
        config.channels = 1  # number of features

        model = AutoModel.from_config(config, predict_sequence_length=8)
        trainer = KerasTrainer(
            model,
        )

        trainer.train(train, valid, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1)

        y_test = trainer.predict(valid[0])
        self.assertEqual(y_test.shape, valid[1].shape)


if __name__ == "__main__":
    unittest.main()
