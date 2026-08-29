import unittest

import numpy as np

from tfts.structure import from_grid, from_knn, laplacian, symmetric_normalize


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


if __name__ == "__main__":
    unittest.main()
