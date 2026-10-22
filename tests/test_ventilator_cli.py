"""CLI regressions for ventilator training, resume, and checkpoint verification."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from benchmark.datasets.ventilator import VentilatorDataset
from benchmark.verify_saved_models import _build_model
from tests.test_ventilator import ventilator_frame


class VentilatorCommandTest(unittest.TestCase):
    """Exercise training, checkpoint recovery, and verification through the CLI."""

    def run_module(self, module, *args):
        result = subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_resume_recovers_predictions_from_original_protocol(self):
        # Given a complete small run with nondefault training and architecture settings.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.csv"
            ventilator_frame().to_csv(data, index=False)
            output = root / "run"
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "benchmark": {
                            "epochs": 1,
                            "batch_size": 2,
                            "learning_rate": 0.002,
                            "seed": 17,
                            "verbose": 0,
                            "output_dir": str(output),
                        },
                        "models": {"rnn": {"rnn_hidden_size": 2, "dense_hidden_size": 4}},
                    }
                )
            )
            self.run_module("benchmark.ventilator", "--data", data, "--config", config, "--threads", 1)
            original = np.load(output / "rnn_oof.npy")
            retained = (output / "rnn_fold0.weights.h5").read_bytes()
            (output / "rnn_fold4.weights.h5").unlink()
            np.save(output / "rnn_oof.npy", np.full_like(original, np.nan))
            # When resuming after a missing checkpoint and corrupt OOF snapshot.
            self.run_module("benchmark.resume_aligned_final", "--data", data, "--output", output, "--threads", 1)
            # Then the same protocol rebuilds the predictions and preserves completed weights.
            np.testing.assert_allclose(np.load(output / "rnn_oof.npy"), original, atol=1e-6)
            self.assertEqual((output / "rnn_fold0.weights.h5").read_bytes(), retained)
            (output / "folds.npz").unlink()
            verified = self.run_module(
                "benchmark.verify_saved_models", "--data", data, "--output", output, "--threads", 1
            )
            self.assertIn("result: PASS", verified.stdout)

    def test_transformer_cli_honors_configuration(self):
        # Given nondefault model sizes and a one-epoch learning-rate budget.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.csv"
            ventilator_frame().to_csv(data, index=False)
            output = root / "run"
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "benchmark": {
                            "epochs": 1,
                            "batch_size": 2,
                            "learning_rate": 0.0002,
                            "verbose": 0,
                            "output_dir": str(output),
                        },
                        "models": {
                            "reference": {"num_blocks": 1, "embed_dim": 4, "num_heads": 2, "ff_dim": 8},
                            "bert": {
                                "num_layers": 1,
                                "hidden_size": 8,
                                "num_attention_heads": 2,
                                "ffn_intermediate_size": 8,
                            },
                        },
                    }
                )
            )
            # When both models train through the real comparison command.
            self.run_module(
                "benchmark.ventilator_bert", "--data", data, "--config", config, "--threads", 1, "--folds", 1
            )
            # Then saved models reload with the requested architecture and learning rate.
            dataset = VentilatorDataset()
            x, _, _ = dataset.prepare_reference(data)
            _, valid = dataset.fold_indices(0)
            for name in ("reference", "bert"):
                history = json.loads((output / f"{name}_fold0_history.json").read_text())
                self.assertEqual(len(history["loss"]), 1)
                self.assertIsInstance(history["loss"][0], float)
                self.assertIsInstance(history["learning_rate"][0], float)
                self.assertAlmostEqual(history["learning_rate"][0], 0.0002)
                model = _build_model(name, x.shape[-1], output)
                model(x[valid])
                model.load_weights(str(output / f"{name}_fold0.weights.h5"))
                np.testing.assert_allclose(model(x[valid]), np.load(output / f"{name}_oof.npy")[valid], atol=1e-5)
            self.assertEqual(json.loads((output / "results.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()
