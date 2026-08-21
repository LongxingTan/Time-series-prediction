import unittest

import numpy as np
import tensorflow as tf

from tfts.generation import ForecastGenerationConfig
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.models.deep_ar import DeepAR


class DeepARTest(unittest.TestCase):
    def setUp(self):
        self.predict_length = 8
        self.inputs = {
            "x": tf.random.normal([2, 12, 1]),
            "decoder_feature": tf.random.normal([2, self.predict_length, 1]),
            "static": tf.constant([[0], [1]], dtype=tf.int32),
        }

    def test_model(self):
        model = DeepAR(predict_sequence_length=self.predict_length)
        output = model(self.inputs)

        self.assertEqual(output["loc"].shape, (2, self.predict_length, 1))
        self.assertEqual(output["scale"].shape, (2, self.predict_length, 1))
        self.assertTrue(bool(tf.reduce_all(output["scale"] > 0)))

    def test_generation_is_reproducible_and_chunk_independent(self):
        model = DeepAR(predict_sequence_length=self.predict_length)
        model(self.inputs)
        generation_inputs = {"x": self.inputs["x"], "static": self.inputs["static"]}
        config = ForecastGenerationConfig(num_samples=4, return_samples=True, seed=7)

        output = model.generate(generation_inputs, config)
        chunked = model.generate(generation_inputs, config, batch_size=1)

        self.assertEqual(output.predictions.shape, (2, self.predict_length, 1))
        self.assertEqual(output.samples.shape, (2, 4, self.predict_length, 1))
        np.testing.assert_allclose(output.samples.numpy(), chunked.samples.numpy(), rtol=1e-6, atol=1e-6)

    def test_greedy_and_teacher_forced_shapes(self):
        model = DeepAR(predict_sequence_length=self.predict_length)
        forward = model(self.inputs)

        greedy = model.generate(self.inputs, {"mode": "greedy"})
        teacher = model.generate(self.inputs, {"mode": "teacher_forced"})

        self.assertEqual(greedy.predictions.shape, (2, self.predict_length, 1))
        self.assertEqual(teacher.predictions.shape, (2, self.predict_length, 1))
        self.assertEqual(teacher.loc.shape, (2, 1, self.predict_length, 1))
        np.testing.assert_allclose(teacher.predictions.numpy(), forward["loc"].numpy(), rtol=1e-5, atol=1e-5)
        self.assertEqual(len(model.output_distribution.trainable_weights), 4)

    def test_even_sample_median(self):
        trajectories = tf.constant([[[[1.0]], [[3.0]], [[7.0]], [[9.0]]]])
        median = DeepAR._aggregate(trajectories, "median")
        np.testing.assert_allclose(median.numpy(), [[[5.0]]])

    def test_auto_model_delegates_generation_after_build(self):
        config = AutoConfig.for_model("deep_ar")
        model = AutoModel.from_config(config, predict_sequence_length=self.predict_length)
        symbolic_inputs = {
            "x": tf.keras.Input(shape=(12, 1), name="x"),
            "decoder_feature": tf.keras.Input(shape=(self.predict_length, 1), name="decoder_feature"),
            "static": tf.keras.Input(shape=(1,), dtype="int32", name="static"),
        }
        model.build_model(symbolic_inputs)
        output = model.generate(self.inputs, {"mode": "greedy"})

        self.assertEqual(output.predictions.shape, (2, self.predict_length, 1))
