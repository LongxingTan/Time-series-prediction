import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, GridStructure, SpatialLayout, TimeSeriesBatch


class BatchStructureTest(unittest.TestCase):
    def test_plain_batch_remains_rank_three(self):
        batch = TimeSeriesBatch(tf.zeros([2, 8, 1]))
        self.assertEqual(batch.layout, SpatialLayout.NONE)
        self.assertEqual(batch.spatial_axes, ())
        with self.assertRaisesRegex(ValueError, "none layout expects rank-3"):
            TimeSeriesBatch(tf.zeros([2, 8, 3, 1]))

    def test_graph_and_grid_dimensions_are_validated(self):
        graph = TimeSeriesBatch(tf.zeros([2, 8, 3, 1]), structure=GraphStructure(3, adjacency=tf.eye(3)))
        grid = TimeSeriesBatch(tf.zeros([2, 8, 3, 5, 1]), structure=GridStructure(3, 5))
        self.assertEqual(graph.spatial_axes, (2,))
        self.assertEqual(grid.spatial_axes, (2, 3))
        with self.assertRaisesRegex(ValueError, "declares 4"):
            TimeSeriesBatch(tf.zeros([2, 8, 3, 1]), structure=GraphStructure(4))

    def test_spatial_future_values_and_shared_covariates_are_supported(self):
        batch = TimeSeriesBatch(
            tf.zeros([2, 8, 3, 1]),
            future_values=tf.zeros([2, 4, 3, 1]),
            past_time_features=tf.zeros([2, 8, 2]),
            structure=GraphStructure(3, adjacency=tf.eye(3)),
        )
        batch.validate_for("forecasting")

    def test_tensor_dictionary_round_trip(self):
        original = TimeSeriesBatch(tf.zeros([2, 8, 3, 1]), structure=GraphStructure(3, adjacency=tf.eye(3)))
        restored = TimeSeriesBatch.from_inputs(original.as_tensor_dict())
        self.assertEqual(restored.layout, SpatialLayout.NODES)
        self.assertEqual(restored.structure.num_nodes, 3)


if __name__ == "__main__":
    unittest.main()
