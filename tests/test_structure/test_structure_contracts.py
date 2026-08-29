import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, GridStructure, SpatialLayout
from tfts.contracts.structure import SpatialStructure


class StructureContractTest(unittest.TestCase):
    def test_layout_and_base_contract(self):
        self.assertEqual(SpatialLayout.normalize("NODES"), SpatialLayout.NODES)
        self.assertEqual(SpatialLayout.normalize(SpatialLayout.GRID), SpatialLayout.GRID)
        base = SpatialStructure()
        with self.assertRaises(NotImplementedError):
            _ = base.layout
        with self.assertRaises(NotImplementedError):
            _ = base.spatial_shape
        with self.assertRaises(NotImplementedError):
            base.validate(tf.zeros([1, 1, 1]))

    def test_graph_properties_and_tensor_round_trips(self):
        graph = GraphStructure(
            3,
            adjacency=tf.eye(3),
            edge_index=tf.constant([[0, 1], [1, 2]]),
            node_features=tf.ones([3, 2]),
            node_coordinates=tf.ones([3, 2]),
            node_mask=tf.ones([3]),
            node_ids=(1, 2, 3),
        )
        graph.validate(tf.zeros([2, 4, 3, 1]))
        self.assertTrue(graph.is_shared)
        self.assertFalse(graph.is_dynamic)
        self.assertEqual(graph.set_size, 3)
        self.assertEqual(graph.node_ids, ("1", "2", "3"))
        tensors = graph.to_tensor_dict("g.")
        self.assertNotIn("g.node_ids", tensors)
        restored = GraphStructure.from_tensor_dict({"structure.graph.adjacency": tf.eye(3)})
        self.assertEqual(restored.num_nodes, 3)
        restored = GraphStructure.from_tensor_dict({"structure.graph.node_mask": tf.ones([4])})
        self.assertEqual(restored.num_nodes, 4)

        dynamic = GraphStructure(3, adjacency=tf.zeros([2, 4, 3, 3]))
        self.assertTrue(dynamic.is_dynamic)
        with self.assertRaisesRegex(ValueError, "cannot be inferred"):
            GraphStructure.from_tensor_dict({"structure.graph.edge_weight": tf.ones([2])})

    def test_graph_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GraphStructure(0)
        with self.assertRaisesRegex(ValueError, "node_ids"):
            GraphStructure(2, node_ids=("only-one",))
        cases = (
            (GraphStructure(2), tf.zeros([1, 2, 1]), "nodes layout"),
            (GraphStructure(2, adjacency=tf.zeros([2, 2, 2, 2, 2])), tf.zeros([1, 2, 2, 1]), "adjacency"),
            (GraphStructure(2, adjacency=tf.zeros([3, 3])), tf.zeros([1, 2, 2, 1]), "trailing"),
            (GraphStructure(2, edge_index=tf.ones([2, 1], tf.float32)), tf.zeros([1, 2, 2, 1]), "integer"),
            (GraphStructure(2, edge_index=tf.ones([1, 3], tf.int32)), tf.zeros([1, 2, 2, 1]), "edge_index"),
            (GraphStructure(2, node_mask=tf.ones([1, 1, 1, 1])), tf.zeros([1, 2, 2, 1]), "node_mask"),
            (GraphStructure(2, node_features=tf.ones([1, 1, 1, 1])), tf.zeros([1, 2, 2, 1]), "node_features"),
            (GraphStructure(2, node_coordinates=tf.ones([1])), tf.zeros([1, 2, 2, 1]), "node_coordinates"),
        )
        for structure, values, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                structure.validate(values)

    def test_grid_validation_and_tensor_round_trips(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GridStructure(0, 2)
        with self.assertRaisesRegex(ValueError, "periodic_axes"):
            GridStructure(2, 3, periodic_axes=("time",))

        grid = GridStructure(
            2, 3, coordinates=tf.zeros([2, 3, 2]), valid_mask=tf.ones([2, 3]), periodic_axes=("width",)
        )
        grid.validate(tf.zeros([1, 4, 2, 3, 1]))
        self.assertEqual(grid.set_size, 6)
        self.assertEqual(GridStructure.from_tensor_dict(grid.to_tensor_dict("structure.grid.")).spatial_shape, (2, 3))
        masked = GridStructure.from_tensor_dict({"structure.grid.valid_mask": tf.ones([4, 5])})
        self.assertEqual(masked.spatial_shape, (4, 5))
        with self.assertRaisesRegex(ValueError, "cannot be inferred"):
            GridStructure.from_tensor_dict({})

        cases = (
            (grid, tf.zeros([1, 4, 2, 3]), "grid layout"),
            (grid, tf.zeros([1, 4, 4, 3, 1]), "height"),
            (grid, tf.zeros([1, 4, 2, 4, 1]), "width"),
            (GridStructure(2, 3, valid_mask=tf.ones([1])), tf.zeros([1, 4, 2, 3, 1]), "valid_mask"),
        )
        for structure, values, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                structure.validate(values)


if __name__ == "__main__":
    unittest.main()
