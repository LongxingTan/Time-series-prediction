import tempfile
import unittest

import numpy as np


def _synthetic_windows(n=2000, train_length=12, seed=0):
    """Deterministic, slightly nonstationary signal to reconstruct."""
    rng = np.random.default_rng(seed)
    base = np.sin(np.arange(0, n + train_length) / 5.0) + 0.05 * rng.normal(size=n + train_length)
    from examples.run_anomaly import create_subsequences

    return create_subsequences(base.reshape(-1, 1), train_length)


class AnomalyExampleTest(unittest.TestCase):
    def test_create_subsequences_shape(self):
        from examples.run_anomaly import create_subsequences

        windows = create_subsequences(np.ones((1000, 1)), 12)
        self.assertEqual(windows.shape, (1000 - 12 + 1, 12, 1))

    def test_train_calibrate_detect(self):
        """Exercise the reconstruction-based anomaly flow with synthetic data."""
        from examples.run_anomaly import build_model, perform_inference, train_model
        from tfts import set_seed

        windows = _synthetic_windows()
        split = int(len(windows) * 0.8)
        fit_windows, test_windows = windows[:split], windows[split:]

        args = type(
            "args",
            (),
            {
                "seed": 315,
                "use_model": "tcn",
                "train_length": 12,
                "epochs": 1,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "output_dir": tempfile.mkdtemp(),
            },
        )()

        set_seed(args.seed)
        model = build_model(args)
        train_model(args, model, fit_windows)

        scores, test_windows_out = perform_inference(model, fit_windows, test_windows)
        self.assertEqual(scores.shape[0], test_windows_out.shape[0])
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
