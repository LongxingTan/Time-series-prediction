import unittest

import numpy as np
import tensorflow as tf

from tfts import (
    AutoConfig,
    AutoModel,
    AutoModelForAnomaly,
    AutoModelForClassification,
    AutoModelForPrediction,
    AutoModelForSegmentation,
    AutoModelForUncertainty,
)


class TestAutoModel(unittest.TestCase):
    def test_auto_model_init(
        self,
    ):
        config = AutoConfig.for_model("seq2seq")

        auto_model = AutoModel.from_config(config, predict_sequence_length=5)
        input_data = np.random.rand(1, 10, 3)
        output = auto_model(input_data)

        self.assertEqual(output.shape, (1, 5, 1))

    def test_auto_model_for_prediction(self):
        config = AutoConfig.for_model("bert")
        config.skip_connect_circle = False
        config.skip_connect_mean = False
        model = AutoModelForPrediction.from_config(config, predict_sequence_length=3)

        x = tf.random.normal([2, 14, 4])
        y = model(x)
        self.assertEqual(y.shape, (2, 3, 1))

    def test_auto_model_for_classification(self):
        num_labels = 3
        config = AutoConfig.for_model("bert")
        model = AutoModelForClassification.from_config(config, num_labels=num_labels)

        x = tf.random.normal([2, 14, 4])
        y = model(x)
        self.assertEqual(y.shape, (2, num_labels))

    def test_auto_model_for_anomaly(self):
        config = AutoConfig.for_model("bert")
        config.train_sequence_length = 14
        model = AutoModelForAnomaly.from_config(config)

        x = tf.random.normal([2, 14, 4])
        dist = model.detect(x)
        print(dist.shape)

    def test_auto_model_for_segmentation(self):
        config = AutoConfig.for_model("bert")
        model = AutoModelForSegmentation.from_config(config)

        x = tf.random.normal([2, 14, 4])
        output = model(x)
        print(output.shape)

    def test_auto_model_for_uncertainty(self):
        config = AutoConfig.for_model("bert")
        model = AutoModelForUncertainty.from_config(config)

        x = tf.random.normal([2, 14, 4])
        output = model(x)
        print(output.shape)
