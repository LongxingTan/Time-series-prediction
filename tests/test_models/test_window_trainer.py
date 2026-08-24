import unittest

import numpy as np
import tensorflow as tf

from tfts.models.timemixer import TimeMixer, TimeMixerConfig
from tfts.models.timesnet import TimesNet, TimesNetConfig
from tfts.models.timexer import TimeXer, TimeXerConfig
from tfts.training import WindowedTrainer


class WindowedTrainerTest(unittest.TestCase):
    def _synthetic(self, n=8):
        np.random.seed(0)
        seq, pred = 12, 6
        series = [(np.cumsum(np.random.randn(400) * 0.1) + 100.0).astype(np.float32) for _ in range(n)]
        histories = [s[:-pred] for s in series]
        targets = np.stack([s[-pred:] for s in series]).astype(np.float32)
        return histories, targets, seq, pred

    def test_all_raw_tf_ports_train_and_evaluate(self):
        for name, model_cls, cfg_cls in [
            ("timesnet", TimesNet, TimesNetConfig()),
            ("timemixer", TimeMixer, TimeMixerConfig()),
            ("timexer", TimeXer, TimeXerConfig()),
        ]:
            with self.subTest(model=name):
                histories, targets, seq, pred = self._synthetic()
                model = model_cls(predict_sequence_length=pred, config=cfg_cls)
                trainer = WindowedTrainer(
                    model, seq_len=seq, pred_len=pred, epochs=2, lr=1e-3, batch_size=4, patience=1, seed=7
                )
                logs = trainer.train(histories, targets, verbose=False)
                self.assertEqual(len(logs), 2)
                ev = trainer.evaluate(histories, targets)
                self.assertIn("smape", ev)
                self.assertIn("mae", ev)
                self.assertIn("mse", ev)
                x = np.stack([s[-seq:] for s in histories]).astype(np.float32)[..., None]
                out = trainer.predict(x)
                self.assertEqual(out.shape, (len(histories), pred, 1))

    def test_windows_are_not_constant(self):
        # Regression: a 1D series must be expanded to (L, 1) — otherwise the
        # `[..., :1]` slice collapses each window to its first value.
        from tfts.training.window_trainer import final_windows, sampled_windows

        np.random.seed(1)
        histories = [np.cumsum(np.random.randn(60)) for _ in range(5)]
        x, y, m = sampled_windows(histories, np.random.default_rng(0), 26, 13)
        vals, mask = final_windows(histories, 26)
        for arr in (x, vals):  # each window must contain more than one distinct value
            self.assertGreater(arr.reshape(len(histories), -1).std(axis=1).min(), 0.0)
        self.assertGreater(y.std(), 0.0)


if __name__ == "__main__":
    unittest.main()
