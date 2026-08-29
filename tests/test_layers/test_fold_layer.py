import unittest

import numpy as np
import tensorflow as tf

from tfts.layers import FoldSpatialToBatch, UnfoldBatchToSpatial


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


if __name__ == "__main__":
    unittest.main()
