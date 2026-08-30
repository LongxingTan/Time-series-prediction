import unittest

import tensorflow as tf

from tfts.contracts import GraphStructure, SpatialArrangement, TimeSeriesBatch


class BatchStructureTest(unittest.TestCase):
    def test_rank_declares_arrangement(self):
        batch = TimeSeriesBatch(tf.zeros([2, 8, 1]))
        self.assertEqual(batch.arrangement, SpatialArrangement.NONE)
        self.assertEqual(batch.spatial_axes, ())
        node_set = TimeSeriesBatch(tf.zeros([2, 8, 3, 1]))
        self.assertEqual(node_set.arrangement, SpatialArrangement.SET)
        grid = TimeSeriesBatch(tf.zeros([2, 8, 3, 5, 1]))
        self.assertEqual(grid.arrangement, SpatialArrangement.GRID)
        self.assertEqual(grid.spatial_axes, (2, 3))

    def test_graph_dimensions_are_validated(self):
        graph = TimeSeriesBatch(tf.zeros([2, 8, 3, 1]), structure=GraphStructure(3, adjacency=tf.eye(3)))
        self.assertEqual(graph.spatial_axes, (2,))
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

    def test_tensor_dictionary_round_trip_preserves_graph_metadata(self):
        original = TimeSeriesBatch(
            tf.zeros([2, 8, 3, 1]),
            structure=GraphStructure(3, adjacency=tf.eye(3), node_ids=("a", "b", "c")),
        )
        restored = TimeSeriesBatch.from_inputs(original.as_tensor_dict())
        self.assertEqual(restored.arrangement, SpatialArrangement.SET)
        self.assertEqual(restored.structure.num_nodes, 3)
        self.assertEqual(restored.structure.node_ids, ("a", "b", "c"))

    def test_mapping_boundaries_and_properties(self):
        graph = GraphStructure(2, adjacency=tf.eye(2))
        batch = TimeSeriesBatch(
            tf.zeros([3, 5, 2, 4]), labels=tf.ones([3]), metadata={"source": "test"}, structure=graph
        )
        self.assertEqual(tuple(batch.spatial_shape), (2,))
        self.assertEqual(int(batch.batch_size), 3)
        self.assertEqual(int(batch.context_length), 5)
        self.assertEqual(int(batch.target_dim), 4)
        self.assertIn("metadata", batch.as_dict())
        self.assertNotIn("metadata", batch.as_tensor_dict())
        self.assertIn("future_values", batch.as_dict(include_none=True))
        self.assertNotIn("structure.graph.adjacency", batch.as_tensor_dict(include_structure=False))
        self.assertIs(TimeSeriesBatch.from_inputs(batch), batch)

        with self.assertRaisesRegex(ValueError, "recognized prefix"):
            TimeSeriesBatch.from_inputs(
                {"past_values": tf.zeros([1, 2, 2, 1]), "structure.unknown.value": tf.ones([1])}
            )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            TimeSeriesBatch.from_inputs({"past_values": tf.zeros([1, 2, 1]), "typo": tf.ones([1])})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            TimeSeriesBatch.from_inputs((tf.zeros([1, 2, 1]), tf.zeros([1, 1, 1])))
        with self.assertRaisesRegex(ValueError, "required"):
            TimeSeriesBatch(None)

    def test_validate_for_rejects_misaligned_fields(self):
        graph = GraphStructure(3, adjacency=tf.eye(3))
        cases = (
            ({"past_time_features": tf.zeros([2, 8])}, "past_time_features"),
            ({"past_time_features": tf.zeros([1, 8, 2])}, "batch size"),
            ({"past_time_features": tf.zeros([2, 7, 2])}, "time length"),
            ({"future_values": tf.zeros([2, 4, 3, 1]), "future_time_features": tf.zeros([2, 5, 1])}, "horizon"),
            ({"past_observed_mask": tf.zeros([2, 8, 3, 2])}, "same shape"),
            ({"future_values": tf.zeros([2, 4, 3, 1]), "future_observed_mask": tf.zeros([2, 4, 3, 2])}, "same shape"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                (ValueError, tf.errors.InvalidArgumentError), message
            ):
                TimeSeriesBatch(tf.zeros([2, 8, 3, 1]), structure=graph, **kwargs).validate_for("forecasting")

        with self.assertRaisesRegex(ValueError, "imputation requires"):
            TimeSeriesBatch(tf.zeros([2, 8, 1])).validate_for("imputation")
        with self.assertRaisesRegex(ValueError, "classification labels"):
            TimeSeriesBatch(tf.zeros([2, 8, 1]), labels=tf.zeros([2, 1, 1])).validate_for("classification")


if __name__ == "__main__":
    unittest.main()
