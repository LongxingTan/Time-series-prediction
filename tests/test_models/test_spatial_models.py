import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting, GraphStructure, TimeSeriesBatch
from tfts.saving import load_model


class SpatialModelTest(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(7)
        self.values = tf.random.normal([2, 8, 4, 2])
        self.batch = TimeSeriesBatch(
            self.values,
            structure=GraphStructure(4, adjacency=tf.eye(4)),
        )

    def test_plain_model_directive_and_per_node_fallback(self):
        config = AutoConfig.for_model("dlinear")
        with self.assertRaisesRegex(ValueError, "spatial_strategy='per_node'"):
            AutoModelForForecasting.from_config(config, prediction_length=3)(self.batch)
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("dlinear"), prediction_length=3, spatial_strategy="per_node"
        )
        result = model(self.batch)
        self.assertEqual(result.shape, (2, 3, 4, 1))

    def test_per_node_matches_independent_node_at_inference(self):
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("dlinear"), prediction_length=3, spatial_strategy="per_node"
        )
        combined = model(self.batch, training=False)
        node = TimeSeriesBatch(self.values[:, :, 1, :])
        independent = model.backbone(node.past_values)
        np.testing.assert_allclose(combined[:, :, 1, :].numpy(), independent.numpy(), atol=1e-6)

    def test_stgcn_eager_and_traced_shapes(self):
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("stgcn"), prediction_length=3)
        self.assertEqual(model(self.batch).shape, (2, 3, 4, 1))

        @tf.function
        def run(inputs):
            return model(inputs)

        self.assertEqual(run(self.batch.as_tensor_dict()).shape, (2, 3, 4, 1))

    def test_spatial_keras_round_trip(self):
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("stgcn"), prediction_length=3)
        expected = model(self.batch)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stgcn.keras")
            model.save(path)
            restored = load_model(path, compile=False)
            actual = restored(self.batch)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
