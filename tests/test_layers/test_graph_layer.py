import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.graph_layer import AdjacencyPolynomialConv, GraphAttention, GraphConv


class GraphLayerTest(unittest.TestCase):
    def test_chebyshev_convolution_orders_and_config(self):
        features = tf.ones([2, 3, 2])
        adjacency = tf.eye(3)
        layer = AdjacencyPolynomialConv(4, k=3, activation="relu", use_bias=False)
        output = layer((features, adjacency))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(layer.get_config()["k"], 3)
        self.assertFalse(layer.get_config()["use_bias"])
        self.assertTrue(layer.get_config()["normalize"])

        first_order = AdjacencyPolynomialConv(2, k=1)((features, adjacency))
        self.assertEqual(first_order.shape, features.shape)
        with self.assertRaisesRegex(ValueError, "at least one"):
            AdjacencyPolynomialConv(2, k=0)

    def test_polynomial_convolution_normalizes_unnormalized_adjacency(self):
        features = tf.ones([1, 3, 1])
        adjacency = tf.constant([[0.0, 10.0, 0.0], [10.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
        layer = AdjacencyPolynomialConv(2, k=5, normalize=True)
        output = layer((features, adjacency))
        self.assertTrue(np.all(np.isfinite(output.numpy())))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            layer((features, -tf.eye(3)))

    def test_graph_convolution_layer(self):
        units = 32
        batch_size = 2
        num_nodes = 10
        input_dim = 5

        # Test Dense Adjacency
        layer = GraphConv(units, activation="relu")

        # Inputs: Features (B, N, F), Adjacency (B, N, N)
        x = tf.random.normal((batch_size, num_nodes, input_dim))
        a = tf.random.uniform((batch_size, num_nodes, num_nodes))

        y = layer((x, a))

        # Output shape should be (B, N, Units)
        self.assertEqual(y.shape, (batch_size, num_nodes, units))

        # Test Config
        config = layer.get_config()
        self.assertEqual(config["units"], units)
        self.assertEqual(config["use_bias"], True)

    def test_graph_convolution_sparse(self):
        # Test with Sparse Tensor Adjacency (Single graph mode usually)
        units = 16
        num_nodes = 50
        input_dim = 8

        layer = GraphConv(units)

        # Features (1, N, F) - usually sparse matmul requires specific dimensions
        # Here we test the mechanics of passing a SparseTensor
        x = tf.random.normal((num_nodes, input_dim))

        # Create a random sparse adjacency matrix
        indices = []
        values = []
        for i in range(num_nodes):
            indices.append([i, (i + 1) % num_nodes])
            values.append(1.0)

        a_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[num_nodes, num_nodes])

        # Note: The layer logic handles matmul.
        # If the input x is rank 2 (N, F), output is (N, Units)
        y = layer((x, a_sparse))
        self.assertEqual(y.shape, (num_nodes, units))

        no_bias = GraphConv(3, use_bias=False)
        self.assertEqual(no_bias((x, tf.eye(num_nodes))).shape, (num_nodes, 3))
        self.assertEqual(no_bias.compute_output_shape(((None, input_dim), (num_nodes, num_nodes))), (None, 3))

    def test_graph_attention_layer_concat(self):
        units = 8
        num_heads = 4
        batch_size = 2
        num_nodes = 10
        input_dim = 5

        # Test head_reduction='concat'
        layer = GraphAttention(units=units, num_heads=num_heads, head_reduction="concat", activation="relu")

        x = tf.random.normal((batch_size, num_nodes, input_dim))
        a = tf.random.uniform((batch_size, num_nodes, num_nodes))

        y = layer((x, a), training=True)

        # Expected shape: (B, N, units * num_heads)
        self.assertEqual(y.shape, (batch_size, num_nodes, units * num_heads))

        config = layer.get_config()
        self.assertEqual(config["num_heads"], num_heads)
        self.assertEqual(config["head_reduction"], "concat")

    def test_graph_attention_layer_average(self):
        units = 16
        num_heads = 2
        batch_size = 2
        num_nodes = 10
        input_dim = 5

        # Test head_reduction='average'
        layer = GraphAttention(units=units, num_heads=num_heads, head_reduction="average")

        x = tf.random.normal((batch_size, num_nodes, input_dim))
        a = tf.random.uniform((batch_size, num_nodes, num_nodes))

        y = layer((x, a))

        # Expected shape: (B, N, units)
        self.assertEqual(y.shape, (batch_size, num_nodes, units))
        self.assertEqual(
            layer.compute_output_shape(((None, num_nodes, input_dim), (None, num_nodes, num_nodes))),
            (None, num_nodes, units),
        )

    def test_graph_attention_rejects_unknown_reduction(self):
        with self.assertRaisesRegex(ValueError, "concat, average"):
            GraphAttention(4, head_reduction="sum")


if __name__ == "__main__":
    unittest.main()
