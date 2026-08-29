import unittest

import numpy as np
import tensorflow as tf

from tfts.contracts import GraphStructure, TimeSeriesBatch
from tfts.layers import FoldSpatialToBatch, UnfoldBatchToSpatial
from tfts.layers.fold_layer import SpatialBatchTransform


class FoldLayerTest(unittest.TestCase):
    def test_node_and_grid_round_trip(self):
        for shape, spatial_shape in (((2, 5, 3, 2), (3,)), ((2, 5, 3, 4, 2), (3, 4))):
            with self.subTest(shape=shape):
                values = tf.reshape(tf.range(np.prod(shape), dtype=tf.float32), shape)
                folded = FoldSpatialToBatch()(values)
                restored = UnfoldBatchToSpatial(spatial_shape)(folded, batch_size=tf.shape(values)[0])
                np.testing.assert_array_equal(restored.numpy(), values.numpy())

    def test_unfold_config_round_trip(self):
        layer = UnfoldBatchToSpatial((3, 4))
        restored = UnfoldBatchToSpatial.from_config(layer.get_config())
        self.assertEqual(restored.spatial_shape, (3, 4))

    def test_identity_inferred_size_and_strategy_validation(self):
        values = tf.ones([2, 4, 3])
        self.assertIs(FoldSpatialToBatch()(values), values)
        self.assertIs(UnfoldBatchToSpatial(())(values), values)

        folded = tf.reshape(tf.range(48, dtype=tf.float32), [6, 4, 2])
        self.assertEqual(UnfoldBatchToSpatial((3,))(folded).shape, (2, 4, 3, 2))
        with self.assertRaisesRegex(ValueError, "spatial_strategy"):
            SpatialBatchTransform("invalid")

    def test_per_node_transform_aligns_temporal_static_and_masks(self):
        batch = TimeSeriesBatch(
            tf.zeros([2, 4, 3, 1]),
            future_values=tf.zeros([2, 2, 3, 1]),
            past_time_features=tf.zeros([2, 4, 2]),
            static_real_features=tf.zeros([2, 5]),
            static_categorical_features=tf.zeros([2, 3, 1]),
            padding_mask=tf.ones([2, 4]),
            labels=tf.ones([2, 2, 3, 1]),
            structure=GraphStructure(3, adjacency=tf.eye(3)),
        )
        transformed, restore = SpatialBatchTransform("per_node").apply(batch)
        self.assertEqual(transformed.past_values.shape, (6, 4, 1))
        self.assertEqual(transformed.future_values.shape, (6, 2, 1))
        self.assertEqual(transformed.past_time_features.shape, (6, 4, 2))
        self.assertEqual(transformed.static_real_features.shape, (6, 5))
        self.assertEqual(transformed.static_categorical_features.shape, (6, 1))
        self.assertEqual(transformed.padding_mask.shape, (6, 4))
        self.assertEqual(transformed.labels.shape, (6, 2, 3, 1))
        self.assertEqual(restore(tf.zeros([6, 2, 1])).shape, (2, 2, 3, 1))
        self.assertIsNone(restore(None))

    def test_flatten_transform_flattens_spatial_features(self):
        batch = TimeSeriesBatch(
            tf.zeros([2, 4, 3, 2]),
            past_time_features=tf.zeros([2, 4, 3]),
            static_real_features=tf.zeros([2, 3, 2]),
            static_categorical_features=tf.zeros([2, 1]),
            labels=tf.ones([2, 1]),
            structure=GraphStructure(3),
        )
        transformed, restore = SpatialBatchTransform("flatten").apply(batch)
        self.assertEqual(transformed.past_values.shape, (2, 4, 6))
        self.assertEqual(transformed.past_time_features.shape, (2, 4, 3))
        self.assertEqual(transformed.static_real_features.shape, (2, 6))
        self.assertEqual(transformed.static_categorical_features.shape, (2, 1))
        self.assertEqual(transformed.labels.shape, (2, 1))
        marker = tf.ones([2, 1, 1])
        self.assertIs(restore(marker), marker)


if __name__ == "__main__":
    unittest.main()
