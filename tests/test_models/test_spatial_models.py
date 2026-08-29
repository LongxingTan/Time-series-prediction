import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting, GraphStructure, TimeSeriesBatch
from tfts.models.stgcn import STGCN, STGCNConfig, _STGCNBlock
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

    def test_stgcn_batched_adjacency_config_and_errors(self):
        batch = TimeSeriesBatch(
            self.values,
            structure=GraphStructure(4, adjacency=tf.repeat(tf.eye(4)[None], 2, axis=0)),
        )
        model = AutoModelForForecasting.from_config(AutoConfig.for_model("stgcn"), prediction_length=2)
        self.assertEqual(model(batch).shape, (2, 2, 4, 1))

        block = _STGCNBlock(8, 2, 3, 0.2)
        self.assertEqual(block.get_config()["temporal_kernel"], 3)
        with self.assertRaisesRegex(ValueError, "cheb_k"):
            STGCNConfig(cheb_k=0)
        with self.assertRaisesRegex(ValueError, "temporal_kernel"):
            STGCNConfig(temporal_kernel=0)

        backbone = STGCN(predict_sequence_length=2, config=STGCNConfig(hidden_size=8, num_layers=1))
        with self.assertRaisesRegex(ValueError, "dense adjacency"):
            backbone.adapt_batch(TimeSeriesBatch(self.values, structure=GraphStructure(4)))
        with self.assertRaisesRegex(ValueError, "time-varying"):
            backbone({"values": self.values, "adjacency": tf.zeros([2, 8, 4, 4])})
        result = backbone({"values": self.values, "adjacency": tf.eye(4)}, return_dict=True)
        self.assertEqual(result["predictions"].shape, (2, 2, 4, 1))

    def test_per_node_learned_forecast_heads_restore_spatial_axes(self):
        config = AutoConfig.for_model("bert")
        config.hidden_size = 8
        config.num_layers = 1
        config.num_attention_heads = 1
        for head, expected in (("point", (2, 3, 4, 1)), ("quantile", (2, 3, 4, 1, 3)), ("distribution", (2, 3, 4, 1))):
            with self.subTest(head=head):
                kwargs = {"prediction_length": 3, "head": head, "spatial_strategy": "per_node"}
                if head == "quantile":
                    kwargs["quantiles"] = (0.1, 0.5, 0.9)
                model = AutoModelForForecasting.from_config(config, **kwargs)
                output = model(self.batch, return_dict=True)
                primary = output.quantile_values if head == "quantile" else output.predictions
                self.assertEqual(primary.shape, expected)
                if head == "distribution":
                    self.assertEqual(output.distribution_params["loc"].shape, expected)

    def test_per_node_native_distribution_restores_parameters(self):
        batch = TimeSeriesBatch(
            self.values[..., :1],
            future_values=tf.zeros([2, 3, 4, 1]),
            static_categorical_features=tf.zeros([2, 1], dtype=tf.int32),
            structure=GraphStructure(4, adjacency=tf.eye(4)),
        )
        model = AutoModelForForecasting.from_config(
            AutoConfig.for_model("deep_ar"), prediction_length=3, spatial_strategy="per_node"
        )
        output = model(batch, return_dict=True)
        self.assertEqual(output.predictions.shape, (2, 3, 4, 1))
        self.assertEqual(output.distribution_params["loc"].shape, (2, 3, 4, 1))

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
