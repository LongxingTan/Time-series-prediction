import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, SpatialArrangement
from tfts.contracts.structure import SpatialStructure


class StructureContractTest(unittest.TestCase):
    def test_arrangement_and_base_contract(self):
        self.assertEqual(SpatialArrangement.normalize("set"), SpatialArrangement.SET)
        self.assertEqual(SpatialArrangement.normalize(SpatialArrangement.GRID), SpatialArrangement.GRID)
        base = SpatialStructure()
        with self.assertRaises(NotImplementedError):
            _ = base.arrangement
        with self.assertRaises(NotImplementedError):
            _ = base.spatial_shape
        with self.assertRaises(NotImplementedError):
            base.validate(tf.zeros([1, 1, 1]))

    def test_graph_properties_and_tensor_round_trip(self):
        graph = GraphStructure(
            3,
            adjacency=tf.eye(3),
            node_features=tf.ones([3, 2]),
            node_mask=tf.ones([3]),
            node_ids=(1, 2, 3),
        )
        graph.validate(tf.zeros([2, 4, 3, 1]))
        self.assertTrue(graph.is_shared)
        self.assertFalse(graph.is_dynamic)
        self.assertEqual(graph.set_size, 3)
        self.assertEqual(graph.node_ids, ("1", "2", "3"))

        tensors = graph.to_tensor_dict()
        self.assertEqual(int(tensors["structure.graph.num_nodes"]), 3)
        self.assertEqual(tensors["structure.graph.node_ids"].dtype, tf.string)
        restored = SpatialStructure.from_tensor_dict(tensors)
        self.assertEqual(restored.num_nodes, 3)
        self.assertEqual(restored.node_ids, graph.node_ids)
        tf.debugging.assert_equal(restored.adjacency, graph.adjacency)

        dynamic = GraphStructure(3, adjacency=tf.zeros([2, 4, 3, 3]))
        self.assertTrue(dynamic.is_dynamic)
        with self.assertRaisesRegex(ValueError, "recognized prefix"):
            SpatialStructure.from_tensor_dict({"structure.unknown.value": tf.ones([2])})

    def test_graph_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GraphStructure(0)
        with self.assertRaisesRegex(ValueError, "node_ids"):
            GraphStructure(2, node_ids=("only-one",))
        cases = (
            (GraphStructure(2), tf.zeros([1, 2, 1]), "nodes layout"),
            (GraphStructure(2, adjacency=tf.zeros([2, 2, 2, 2, 2])), tf.zeros([1, 2, 2, 1]), "adjacency"),
            (GraphStructure(2, adjacency=tf.zeros([3, 3])), tf.zeros([1, 2, 2, 1]), "trailing"),
            (GraphStructure(2, node_mask=tf.ones([1, 1, 1, 1])), tf.zeros([1, 2, 2, 1]), "node_mask"),
            (GraphStructure(2, node_features=tf.ones([1, 1, 1, 1])), tf.zeros([1, 2, 2, 1]), "node_features"),
        )
        for structure, values, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                structure.validate(values)


if __name__ == "__main__":
    unittest.main()
