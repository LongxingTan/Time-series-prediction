"""GPU / multi-GPU example integration tests.

These tests exercise the real TFTS training paths on accelerated hardware and
are skipped automatically on CPU-only runners (e.g. GitHub Actions). Because the
repo's ``make test`` intentionally does not restrict ``CUDA_VISIBLE_DEVICES``,
a local machine with one or more GPUs will run these tests too.
"""

import tempfile
import unittest

import tensorflow as tf

N_GPUS = len(tf.config.list_physical_devices("GPU"))

needs_gpu = unittest.skipUnless(N_GPUS >= 1, "requires a GPU")
needs_multigpu = unittest.skipUnless(N_GPUS >= 2, "requires >= 2 GPUs")


def _forecast_args(**overrides):
    defaults = dict(
        seed=315,
        use_data="sine",
        use_model="dlinear",
        train_length=10,
        predict_sequence_length=5,
        epochs=1,
        batch_size=16,
        learning_rate=0.003,
        strategy="auto",
    )
    defaults.update(overrides)
    return type("args", (), defaults)()


class GpuExampleTest(unittest.TestCase):
    @needs_multigpu
    def test_forecast_manual_two_gpus(self):
        """Full forecasting training with strategy='auto' on 2 GPUs (MirroredStrategy)."""
        from examples.run_prediction_simple import run_manual
        from tfts import set_seed

        set_seed(315)
        run_manual(_forecast_args(strategy="auto"))

    @needs_gpu
    def test_anomaly_on_gpu(self):
        """Anomaly reconstruction training + detection runs on a GPU."""
        from examples.run_anomaly import build_model, perform_inference, train_model
        from tests.test_examples.test_anomaly import _synthetic_windows
        from tfts import set_seed

        windows = _synthetic_windows(n=1024)
        split = int(len(windows) * 0.8)
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
        set_seed(315)
        model = build_model(args)
        train_model(args, model, windows[:split])
        scores, _ = perform_inference(model, windows[:split], windows[split:])
        self.assertEqual(scores.shape[0], len(windows) - split)


if __name__ == "__main__":
    unittest.main()
