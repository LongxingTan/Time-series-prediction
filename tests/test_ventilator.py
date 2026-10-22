"""Regression checks for breath isolation and inspiration-only scoring."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd  # noqa: PANDAS_OK - fixtures use the project's declared data dependency
import tensorflow as tf

from benchmark.datasets.ventilator import VentilatorDataError, VentilatorDataset, inspiration_mae
from benchmark.ventilator_bert import _lr_trace, _segments
from benchmark.ventilator_models import ReferenceTransformer, build_lstm_baseline, masked_multi_mae, ventilator_loss
from benchmark.verify_saved_models import _build_model, _load_folds, _load_pipeline, compare_metrics, verify_model
from tfts.models.bert import Bert, BertConfig


class VentilatorTest(unittest.TestCase):
    def test_features_and_folds_preserve_breaths(self):
        # Given complete breaths in reverse row order.
        rows = 7 * 80
        frame = pd.DataFrame(
            {
                "breath_id": np.repeat(np.arange(7), 80),
                "time_step": np.tile(np.arange(80), 7),
                "R": np.full(rows, 20),
                "C": np.full(rows, 50),
                "u_in": np.ones(rows),
                "u_out": np.tile(np.arange(80) >= 40, 7),
                "pressure": np.full(rows, 999),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame.iloc[::-1].to_csv(path, index=False)
            # When data is loaded and split into all five folds.
            dataset = VentilatorDataset()
            x, y = dataset.prepare_data(path=str(path))
            folds = [dataset.fold_indices(fold) for fold in range(5)]
        # Then cumulative inputs reset each breath, targets stay out of inputs,
        # and every breath is held out exactly once.
        np.testing.assert_array_equal(x[:, :, 3], np.tile(np.arange(1, 81), (7, 1)))
        self.assertEqual(x.shape, (7, 80, 5))
        self.assertTrue(np.all(y == 999))
        self.assertFalse(np.any(x == 999))
        for train, valid in folds:
            self.assertFalse(np.intersect1d(train, valid).size)
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(7))

    def test_score_uses_global_inspiration_weights(self):
        # Given unequal inspiration lengths and large expiration errors.
        target = np.zeros((2, 2, 1), dtype=np.float32)
        prediction = np.array([1, 1000, 3, 3], dtype=np.float32).reshape(2, 2, 1)
        mask = np.array([1, 0, 1, 1], dtype=np.float32).reshape(2, 2, 1)
        # When the competition score is computed.
        score = inspiration_mae(target, prediction, mask)
        # Then expiration is excluded and breaths are weighted by scored rows.
        self.assertAlmostEqual(score, 7 / 3)

    def test_loss_preserves_per_breath_reduction(self):
        # Given the same unequal inspiration lengths.
        packed = np.array([0, 1, 0, 0, 0, 1, 0, 1], dtype=np.float32).reshape(2, 2, 2)
        prediction = np.array([1, 1000, 3, 3], dtype=np.float32).reshape(2, 2, 1)
        # When the reference training loss is evaluated.
        losses = ventilator_loss(packed, prediction).numpy()
        # Then each breath has its own denominator.
        np.testing.assert_allclose(losses, [1, 3])

    def test_baseline_outputs_one_pressure_per_step(self):
        # Given a small instance of the same architecture.
        model = build_lstm_baseline(dense_dim=8, lstm_dim=4, logit_dim=8)
        # When two breaths are predicted.
        prediction = model(np.zeros((2, 80, 5), dtype=np.float32))
        # Then each input row gets one finite pressure.
        self.assertEqual(prediction.shape, (2, 80, 1))
        self.assertTrue(np.isfinite(prediction.numpy()).all())

    def test_score_rejects_broadcasting(self):
        # Given incompatible shapes that NumPy could silently broadcast.
        target = np.zeros((2, 80, 1), dtype=np.float32)
        # When scoring, then the shape mismatch is rejected.
        with self.assertRaises(VentilatorDataError):
            inspiration_mae(target, target[:, :, 0], np.ones_like(target))


class VentilatorBertTest(unittest.TestCase):
    """Regression checks for the reference transformer vs TFTS BERT comparison."""

    def test_segments_sum_to_epochs_and_restart_trace(self):
        # Given several epoch budgets, the three segments always cover the budget.
        for epochs in (3, 4, 6, 10, 100):
            segments = _segments(epochs)
            self.assertEqual(sum(segments), epochs)
            self.assertEqual(len(segments), 3)
        trace = _lr_trace(6)
        self.assertEqual(trace.shape, (6,))
        # Cosine-with-restarts peaks at LR_MAX at the start and decays to LR_MIN
        # at the end of each segment.
        self.assertAlmostEqual(trace.max(), 6e-4, places=6)
        self.assertAlmostEqual(trace[-1], 1e-6, places=8)
        self.assertGreater(trace[0], trace[-1])

    def test_short_schedules_use_configured_peak(self):
        # Given pipeline-check budgets shorter than three restarts.
        for epochs in (1, 2):
            with self.subTest(epochs=epochs):
                # When the schedule uses a nondefault peak.
                trace = _lr_trace(epochs, peak=2e-4)
                # Then every scheduled epoch exists and uses that peak.
                np.testing.assert_allclose(trace, np.full(epochs, 2e-4))

    def test_mask_loss_ignores_expiration_rows(self):
        # Given a multi-channel target and a mask over inspiration rows.
        targets = np.zeros((2, 4, 3), dtype=np.float32)
        targets[0, :, 0] = 1.0  # pressure
        mask = np.zeros((2, 4, 1), dtype=np.float32)
        mask[0, :, :] = 1.0  # only the first breath is inspiration
        prediction = np.zeros_like(targets)
        packed = np.concatenate([targets, mask], axis=-1)
        # When the masked MAE is evaluated, expiration rows contribute nothing.
        loss = float(masked_multi_mae(tf.constant(packed), tf.constant(prediction)).numpy())
        # One inspiration breath contributes 4 errors of 1.0 (pressure channel
        # over 4 steps) over 12 weighted slots (4 steps x 3 channels).
        self.assertAlmostEqual(loss, 4.0 / 12.0)

    def test_masked_multi_mae_rejects_target_mask_rank_mismatch(self):
        with self.assertRaises(Exception):
            masked_multi_mae(tf.zeros((2, 4, 3)), tf.zeros((2, 4, 3)))

    def test_reference_config_roundtrip(self):
        # Given a small reference transformer with nondefault dimensions.
        model = ReferenceTransformer(5, num_blocks=1, embed_dim=4, num_heads=2, ff_dim=8)
        x = tf.ones((2, 80, 5))
        expected = model(x)
        # When Keras reconstructs its architecture and weights.
        restored = ReferenceTransformer.from_config(model.get_config())
        restored(x)
        restored.set_weights(model.get_weights())
        # Then inference is preserved.
        np.testing.assert_allclose(restored(x), expected, atol=1e-6)

    def test_aligned_bert_rejects_wrong_horizon(self):
        # Given aligned regression with a three-step output contract.
        model = Bert(3, BertConfig(hidden_size=8, num_layers=1, num_attention_heads=2, regression_head=True))
        # When the input length differs, then the model rejects an incorrect output shape.
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model(tf.ones((2, 8, 5)))

    def test_sparse_engineered_features_remain_bounded(self):
        # Given a nonconstant input whose interquartile range is zero.
        frame = ventilator_frame()
        frame["u_in"] = np.tile(np.r_[np.zeros(79), 1.0], 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame.to_csv(path, index=False)
            # When reference preprocessing scales the sparse features.
            x, _, _ = VentilatorDataset().prepare_reference(path)
        # Then rare nonzero values are not amplified by an epsilon denominator.
        self.assertLess(float(np.abs(x).max()), 100)


def ventilator_frame():
    """Five complete breaths for preprocessing and checkpoint regressions."""
    return pd.DataFrame(
        {
            "id": np.arange(400),
            "breath_id": np.repeat(np.arange(5), 80),
            "time_step": np.tile(np.arange(80) / 100, 5),
            "R": 20,
            "C": 50,
            "u_in": 1.0,
            "u_out": np.tile(np.arange(80) >= 40, 5),
            "pressure": 1.0,
        }
    )


class SavedModelVerifierTest(unittest.TestCase):
    """Regression checks for reloading checkpoints to reproduce recorded metrics."""

    def test_compare_metrics_reproduces_within_tolerance(self):
        self.assertTrue(compare_metrics(0.250000, 0.250000, 1e-4))
        self.assertTrue(compare_metrics(0.250000, 0.250050, 1e-4))
        self.assertFalse(compare_metrics(0.250000, 0.251000, 1e-4))
        self.assertTrue(compare_metrics(0.25, 0.25, 1e-9))
        # A non-positive tolerance cannot reproduce even identical values.
        self.assertFalse(compare_metrics(0.25, 0.25, -1e-9))

    def test_compare_metrics_treats_nan_as_failing(self):
        # A missing recorded metric can never be reproduced.
        self.assertFalse(compare_metrics(float("nan"), 0.25, 1e-4))

    def test_baseline_reconstruction_uses_saved_dimensions(self):
        # Given weights trained with a small baseline architecture.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            overrides = {"dense_dim": 8, "lstm_dim": 2, "logit_dim": 4}
            (output / "manifest.json").write_text(json.dumps({"model_overrides": {"lstm_baseline": overrides}}))
            original = build_lstm_baseline(**overrides)
            weights = output / "lstm_baseline_fold0.weights.h5"
            original.save_weights(weights)
            # When the verifier rebuilds and loads the checkpoint.
            restored = _build_model("lstm_baseline", 5, output)
            restored.load_weights(weights)
            # Then the predictions match.
            x = np.ones((1, 80, 5), dtype=np.float32)
            np.testing.assert_allclose(restored(x), original(x), atol=1e-6)

    def test_pipeline_preserves_dataset_for_missing_fold_archive(self):
        # Given a valid dataset and results with no folds.npz.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "train.csv"
            ventilator_frame().to_csv(path, index=False)
            (output / "results.json").write_text(json.dumps([{"model": "rnn"}]))
            # When preprocessing and deterministic fold recovery run together.
            dataset, x, _, _ = _load_pipeline(output, str(path))
            folds = _load_folds(dataset, output)
            # Then every loaded breath is held out once.
            np.testing.assert_array_equal(np.sort(np.concatenate([valid for _, valid in folds])), np.arange(len(x)))

    def test_verification_rejects_invalid_evidence(self):
        # Given real checkpoints and matching predictions for five folds.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            overrides = {"rnn_hidden_size": 2, "dense_hidden_size": 4}
            (output / "manifest.json").write_text(json.dumps({"model_overrides": {"rnn": overrides}}))
            x = np.ones((5, 80, 5), dtype=np.float32)
            model = _build_model("rnn", 5, output)
            prediction = model(x).numpy()
            targets = prediction + 1
            mask = np.ones_like(targets)
            folds = [(fold, np.array([fold])) for fold in range(5)]
            for fold, _ in folds:
                model.save_weights(str(output / f"rnn_fold{fold}.weights.h5"))
            for case, metrics, saved in (
                ("valid", {"inspiration_mae": 1.0}, prediction),
                ("nan", {"inspiration_mae": 1.0}, np.full_like(prediction, np.nan)),
                ("shape", {"inspiration_mae": 1.0}, prediction[:1]),
                ("missing_metric", {}, prediction + 1),
                ("wrong_metric", {"rmse": 1.0}, prediction),
            ):
                with self.subTest(case=case):
                    (output / "results.json").write_text(json.dumps([{"model": "rnn", "metrics": metrics}]))
                    np.save(output / "rnn_oof.npy", saved)
                    # When verification compares saved evidence to reloaded inference.
                    result = verify_model("rnn", output, x, targets, mask, folds, 1e-4)
                    # Then only complete, finite, matching evidence passes.
                    self.assertEqual(result.ok, case == "valid")


if __name__ == "__main__":
    unittest.main()
