import unittest

import numpy as np
import tensorflow as tf

from tfts.layers.graph_layer import GraphAttention, GraphConvolution


class GraphLayerTest(unittest.TestCase):
    def test_graph_convolution_layer(self):
        units = 32
        batch_size = 2
        num_nodes = 10
        input_dim = 5

        # Test Dense Adjacency
        layer = GraphConvolution(units, activation="relu")

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

        layer = GraphConvolution(units)

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


if __name__ == "__main__":
    unittest.main()
