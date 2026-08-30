import unittest

import numpy as np

from tfts.graph import (
    add_self_loops,
    from_adjacency,
    from_correlation,
    from_grid,
    from_knn,
    from_radius,
    laplacian,
    random_walk_normalize,
    symmetric_normalize,
)


class StructureBuilderTest(unittest.TestCase):
    def test_knn_has_requested_directed_neighbors(self):
        coordinates = np.array([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
        graph = from_knn(coordinates, k=2, symmetric=False)
        counts = np.count_nonzero(np.asarray(graph.adjacency), axis=1)
        np.testing.assert_array_equal(counts, np.full(4, 2))

    def test_grid_degrees_and_periodic_width(self):
        graph = from_grid(4, 4)
        degree = np.sum(np.asarray(graph.adjacency), axis=1)
        self.assertEqual(degree[0], 2)
        self.assertEqual(degree[5], 4)
        periodic = from_grid(4, 4, periodic_axes=("width",))
        periodic_degree = np.sum(np.asarray(periodic.adjacency), axis=1)
        self.assertEqual(periodic_degree[0], 3)

    def test_normalization_and_laplacian(self):
        graph = symmetric_normalize(from_grid(3, 4))
        adjacency = np.asarray(graph.adjacency)
        np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-6)
        eigenvalues = np.linalg.eigvalsh(laplacian(graph))
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-6)
        self.assertLessEqual(float(eigenvalues.max()), 2.0 + 1e-6)

    def test_explicit_adjacency_and_builder_validation(self):
        graph = from_adjacency([[0, 2], [2, 0]], node_ids=("a", "b"))
        self.assertEqual(graph.node_ids, ("a", "b"))
        for matrix, message in (
            ([[0, 1, 0], [1, 0, 1]], "square"),
            ([[0, -1], [1, 0]], "non-negative"),
            ([[0, np.inf], [1, 0]], "finite"),
        ):
            with self.subTest(matrix=matrix), self.assertRaisesRegex(ValueError, message):
                from_adjacency(matrix)

        with self.assertRaisesRegex(ValueError, "between 1"):
            from_knn([[0], [1]], 2)
        with self.assertRaisesRegex(ValueError, "at least two"):
            from_knn([[0]], 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            from_knn([[0], [np.nan]], 1)
        with self.assertRaisesRegex(ValueError, "metric"):
            from_knn([[0], [1]], 1, metric="taxicab")
        with self.assertRaisesRegex(ValueError, "latitude and longitude"):
            from_knn([[0], [1]], 1, metric="haversine")

    def test_distance_and_correlation_builders(self):
        coordinates = [[49.28, -123.12], [49.25, -123.1], [49.0, -123.0]]
        knn = from_knn(coordinates, 1, metric="haversine", symmetric=True)
        np.testing.assert_allclose(knn.adjacency, np.asarray(knn.adjacency).T)
        with self.assertRaisesRegex(ValueError, "sigma"):
            from_knn([[0], [1]], 1, sigma=0)

        radius = from_radius([[0], [1], [4]], radius=1.5)
        self.assertGreater(float(radius.adjacency[0, 1]), 0)
        self.assertEqual(float(radius.adjacency[0, 2]), 0)
        with self.assertRaisesRegex(ValueError, "sigma"):
            from_radius([[0], [1], [4]], radius=1.5, sigma=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            from_radius([[0], [1]], radius=0)

        graph = from_correlation([[1, 1, 3], [2, 2, 2], [3, 3, 1]], threshold=0.9)
        self.assertEqual(float(graph.adjacency[0, 1]), 1.0)
        np.testing.assert_array_equal(np.diag(graph.adjacency), 0)
        with self.assertRaisesRegex(ValueError, "time, node"):
            from_correlation([1, 2, 3])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            from_correlation([[1, 2], [2, 1]], threshold=2)

    def test_grid_options_and_all_transforms(self):
        graph = from_grid(2, 3, connectivity=8, periodic_axes=("height", "width"))
        self.assertEqual(graph.adjacency.shape, (6, 6))
        for args, message in (((0, 2), "positive"), ((2, 2, 6), "4 or 8")):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, message):
                from_grid(*args)
        with self.assertRaisesRegex(ValueError, "periodic_axes"):
            from_grid(2, 2, periodic_axes=("time",))

        isolated = from_adjacency([[0, 0], [0, 0]])
        looped = add_self_loops(isolated, weight=2)
        np.testing.assert_array_equal(looped.adjacency, np.eye(2) * 2)
        np.testing.assert_array_equal(random_walk_normalize(isolated).adjacency, np.zeros((2, 2)))
        np.testing.assert_array_equal(laplacian(from_adjacency([[0, 1], [1, 0]]), normalized=False), [[1, -1], [-1, 1]])

        no_dense = type(isolated)(2)
        with self.assertRaisesRegex(ValueError, "no dense adjacency"):
            add_self_loops(no_dense)
        batched = type(isolated)(2, adjacency=np.zeros((1, 2, 2)))
        with self.assertRaisesRegex(ValueError, "shared"):
            symmetric_normalize(batched)

    def test_distance_kernel_and_threshold_are_configurable(self):
        binary = from_knn([[0], [1], [3]], 1, kernel="binary", symmetric=False)
        np.testing.assert_array_equal(np.asarray(binary.adjacency)[np.asarray(binary.adjacency) > 0], 1.0)
        thresholded = from_radius([[0], [1], [2]], radius=2, sigma=1, epsilon=0.4)
        self.assertEqual(float(thresholded.adjacency[0, 2]), 0.0)
        with self.assertRaisesRegex(ValueError, "kernel"):
            from_knn([[0], [1]], 1, kernel="linear")
        with self.assertRaisesRegex(ValueError, "epsilon"):
            from_radius([[0], [1]], radius=2, epsilon=-1)


if __name__ == "__main__":
    unittest.main()
