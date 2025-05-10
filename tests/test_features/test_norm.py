import unittest

import numpy as np

from tfts.features.norm import denormalize, normalize

EPSILON = 1e-8


class TestNormalization(unittest.TestCase):
    def test_1d_normalization(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        for method in ["standard", "minmax", "robust", "log1p"]:
            with self.subTest(method=method):
                normalized, params = normalize(data, method=method)
                denormalized = denormalize(normalized, params)
                self.assertTrue(
                    np.allclose(data, denormalized, atol=EPSILON), f"Denormalization failed for 1D {method}"
                )

    def test_2d_column_wise_normalization(self):
        data = np.array([[1, 10, 100], [2, 20, 120], [3, 15, 110], [4, 25, 130], [5, 10, 100]], dtype=float)
        for method in ["standard", "minmax", "robust", "log1p"]:
            with self.subTest(method=method):
                normalized, params = normalize(data, method=method, axis=0)
                denormalized = denormalize(normalized, params)
                self.assertTrue(
                    np.allclose(data, denormalized, atol=EPSILON), f"Denormalization failed for 2D column-wise {method}"
                )

    def test_2d_row_wise_normalization(self):
        data = np.array([[1, 2, 3, 4, 50], [10, 20, 30, 40, 50], [5, 5, 5, 5, 5]], dtype=float)
        for method in ["standard", "minmax", "robust"]:
            with self.subTest(method=method):
                normalized, params = normalize(data, method=method, axis=1)
                denormalized = denormalize(normalized, params)
                self.assertTrue(
                    np.allclose(data, denormalized, atol=EPSILON), f"Denormalization failed for 2D row-wise {method}"
                )


if __name__ == "__main__":
    unittest.main()
