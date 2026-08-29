import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, GridStructure, TimeSeriesBatch
from tfts.data import SequenceMaterializer


class SpatialMaterializerTest(unittest.TestCase):
    def test_shared_graph_tensors_are_injected_without_per_sample_copies(self):
        batch = TimeSeriesBatch(
            tf.zeros([3, 4, 2, 1]),
            labels=tf.ones([3, 1]),
            structure=GraphStructure(
                2,
                adjacency=tf.eye(2),
                node_mask=tf.ones([2]),
                node_features=tf.ones([2, 3]),
                node_coordinates=tf.ones([2, 2]),
            ),
        )
        inputs, labels = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2, shuffle=True, seed=7)))
        self.assertEqual(inputs["structure.graph.adjacency"].shape, (2, 2))
        self.assertEqual(inputs["structure.graph.node_mask"].shape, (2,))
        self.assertEqual(inputs["structure.graph.node_features"].shape, (2, 3))
        self.assertEqual(inputs["structure.graph.node_coordinates"].shape, (2, 2))
        self.assertEqual(labels.shape, (2, 1))

    def test_shared_grid_tensors_and_future_exclusion(self):
        batch = TimeSeriesBatch(
            tf.zeros([2, 4, 2, 3, 1]),
            future_values=tf.zeros([2, 1, 2, 3, 1]),
            structure=GridStructure(
                2,
                3,
                valid_mask=tf.ones([2, 3]),
                coordinates=tf.ones([2, 3, 2]),
            ),
        )
        inputs = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2, include_future_values=False)))
        self.assertNotIn("future_values", inputs)
        self.assertEqual(inputs["structure.grid.valid_mask"].shape, (2, 3))
        self.assertEqual(inputs["structure.grid.coordinates"].shape, (2, 3, 2))

    def test_sparse_edge_index_round_trips_through_dataset_boundary(self):
        batch = TimeSeriesBatch(
            tf.zeros([3, 4, 4, 1]),
            structure=GraphStructure(4, edge_index=tf.constant([[0, 1], [1, 2]], tf.int32)),
        )
        inputs = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2)))
        restored = TimeSeriesBatch.from_inputs(inputs)
        self.assertEqual(restored.structure.num_nodes, 4)
        self.assertEqual(restored.structure.edge_index.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
