import unittest

import numpy as np
import tensorflow as tf

from tfts.metrics import mae, mape, mse, smape


class MetricsTest(unittest.TestCase):
    def test_integer_inputs_use_floating_point_arithmetic(self):
        y_true = np.array([0, 2], dtype=np.int32)
        y_pred = np.array([0, 1], dtype=np.int32)

        self.assertTrue(np.isfinite(mape(y_true, y_pred)))
        self.assertTrue(np.isfinite(smape(y_true, y_pred)))
        self.assertAlmostEqual(mse(y_true, y_pred), 0.5)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.5)

    def test_tensor_metrics_are_finite_for_integer_zeros(self):
        y_true = tf.constant([0, 2], dtype=tf.int32)
        y_pred = tf.constant([0, 1], dtype=tf.int32)

        self.assertTrue(bool(tf.math.is_finite(mape(y_true, y_pred))))
        self.assertTrue(bool(tf.math.is_finite(smape(y_true, y_pred))))

    def test_metric_function_is_logged_by_keras_fit(self):
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer="sgd", loss="mse", metrics=[mae])
        history = model.fit(np.ones((2, 1)), np.ones((2, 1)), epochs=1, verbose=0)

        self.assertIn("mae", history.history)


if __name__ == "__main__":
    unittest.main()
