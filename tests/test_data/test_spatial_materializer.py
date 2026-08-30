import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, TimeSeriesBatch
from tfts.data import SequenceMaterializer


class SpatialMaterializerTest(unittest.TestCase):
    def test_shared_graph_tensors_are_injected_without_per_sample_copies(self):
        batch = TimeSeriesBatch(
            tf.zeros([3, 4, 2, 1]),
            labels=tf.ones([3, 1]),
            structure=GraphStructure(2, adjacency=tf.eye(2), node_ids=("north", "south")),
        )
        inputs, labels = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2, shuffle=True, seed=7)))
        self.assertEqual(inputs["structure.graph.adjacency"].shape, (2, 2))
        self.assertEqual(inputs["structure.graph.num_nodes"].shape, ())
        self.assertEqual(inputs["structure.graph.node_ids"].shape, (2,))
        self.assertEqual(labels.shape, (2, 1))

        restored = TimeSeriesBatch.from_inputs(inputs)
        self.assertEqual(restored.structure.num_nodes, 2)
        self.assertEqual(restored.structure.node_ids, ("north", "south"))

    def test_future_values_can_be_excluded(self):
        batch = TimeSeriesBatch(tf.zeros([2, 4, 2, 1]), future_values=tf.zeros([2, 1, 2, 1]))
        inputs = next(iter(SequenceMaterializer.as_tf_dataset(batch, batch_size=2, include_future_values=False)))
        self.assertNotIn("future_values", inputs)


if __name__ == "__main__":
    unittest.main()
