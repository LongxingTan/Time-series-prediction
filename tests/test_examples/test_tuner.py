import unittest
from unittest.mock import patch

import numpy as np


class FakeTrial:
    """Drives suggestions to deterministic small/cheap values."""

    def suggest_int(self, name, low, high, **kwargs):
        return 1  # 1 training epoch keeps the test fast

    def suggest_float(self, name, low, high, **kwargs):
        return low


class TunerExampleTest(unittest.TestCase):
    def setUp(self):
        from examples.run_tuner import AutoTuner
        from tfts import get_data

        (x_train, y_train), (x_valid, y_valid) = get_data("sine", 12, 2, test_size=0.2)
        self.tuner = AutoTuner(
            use_model="rnn",
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            predict_sequence_length=2,
        )

    def test_objective_returns_score(self):
        score = self.tuner.objective(FakeTrial())
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)

    def test_run_returns_study(self):
        with patch.object(self.tuner, "objective", return_value=1.0):
            study = self.tuner.run(n_trials=1, direction="minimize")
        self.assertEqual(study.best_value, 1.0)


if __name__ == "__main__":
    unittest.main()
