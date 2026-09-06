"""Windowed forecasting trainer — the tfts-native training pipeline.

Encapsulates the winning recipe measured on M4 short-term forecasting:

- **elementwise SMAPE loss** with a per-horizon validity mask as ``sample_weight``
  (matches the M4 SMAPE convention),
- **cosine-annealed Adam** learning-rate (default) or constant LR,
- **best-validation snapshot + early stopping**,
- **per-epoch fresh random windows** sampled from each series history,
- held-out scoring on the **final window** of every series.

It works with any tfts model because all tfts models are native Keras models.
Fully self-contained — it does not depend on any ``exps/`` code.
"""

from __future__ import annotations

import tempfile
from typing import List, Sequence

import numpy as np
import tensorflow as tf

from ..data.window_sampling import final_windows, sampled_windows
from ..losses.loss import smape_loss as _smape_loss_fn
from ..models.base import BaseModel

__all__ = ["WindowedTrainer", "final_windows", "sampled_windows", "smape_score"]


def _loss(y_true, y_pred):
    """2-arg Keras-compilable SMAPE (no extra signature params)."""
    return _smape_loss_fn(y_true, y_pred)


def smape_score(pred: np.ndarray, true: np.ndarray) -> float:
    """M4 SMAPE (%), per-series-averaged: ``mean_t 200|p-t|/(|t|+|p|)``."""
    denom = np.abs(true) + np.abs(pred)
    denom[denom == 0.0] = 1.0
    return float((200.0 * np.abs(pred - true) / denom).mean(axis=-1).mean())


# ---------------------------------------------------------------------------
# The public trainer
# ---------------------------------------------------------------------------
class WindowedTrainer:
    """Train a tfts model on windowed ``(lookback -> forecast)`` series.

    Args:
        model: a tfts ``BaseModel`` already constructed with
            ``predict_sequence_length=pred_len`` (e.g. ``TimesNet(predict_sequence_length=pred, ...)``).
        seq_len: lookback window length (default 26).
        pred_len: forecast horizon (default 13).
        num_features: number of input channels per series (default 1).
        lr: peak Adam learning rate (default ``1e-3``).
        batch_size: mini-batch size (default 16).
        epochs: total training epochs / cosine-decay budget (default 40).
        patience: early-stopping patience on validation SMAPE (default 10).
        seed: RNG seed for window sampling and weight init (default 2026).
        lr_schedule: ``'cosine'`` (default) or ``'constant'``.
        jit_compile: whether to compile with ``jit_compile=True``. Keep ``False``
            for TimesNet (its runtime-period reshape is not XLA-compatible).
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        seq_len: int = 26,
        pred_len: int = 13,
        num_features: int = 1,
        lr: float = 1e-3,
        batch_size: int = 16,
        epochs: int = 40,
        patience: int = 10,
        seed: int = 2026,
        lr_schedule: str = "cosine",
        jit_compile: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.lr_schedule = lr_schedule
        self.jit_compile = jit_compile

        if not isinstance(model, tf.keras.Model):
            raise TypeError("WindowedTrainer expects a tf.keras.Model")
        # Build lazily-created state with a real tensor. Compilation happens in
        # ``train()`` once the cosine schedule length is known.
        _ = model(tf.zeros((2, seq_len, num_features), tf.float32))
        self.model = model

    def train(
        self, histories: Sequence[np.ndarray], targets: np.ndarray, val_rng_seed: int = 7, verbose: bool = True
    ) -> List[dict]:
        """Train on windows sampled from ``histories``; validate + early-stop.

        Compiles the wrapped model exactly once with the requested LR schedule,
        then loops: sample fresh windows from every series, fit one SMAPE-masked
        epoch, validate on a fixed window set, keep the best-validation snapshot
        and early-stop on ``patience``. Returns the per-epoch
        ``{epoch, train, val}`` log; the best snapshot is restored afterwards.
        """
        if self.lr_schedule == "cosine":
            # Cosine decay over the full epoch budget (global-step based).
            steps = int(np.ceil(len(histories) / self.batch_size)) * self.epochs
            sched = tf.keras.optimizers.schedules.CosineDecay(self.lr, steps, alpha=0.01)
            optimizer = tf.keras.optimizers.Adam(sched)
        else:
            optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model.compile(optimizer=optimizer, loss=_loss, jit_compile=self.jit_compile)

        del targets, val_rng_seed  # validation is a deterministic temporal holdout
        train_histories = [np.asarray(series)[: -self.pred_len] for series in histories]
        x_val, _ = final_windows(train_histories, self.seq_len, self.num_features)
        y_val, m_val = final_windows(histories, self.pred_len, self.num_features)
        # ``final_windows`` right-aligns short sequences; forecast masks/targets
        # must be left-aligned to match model horizon semantics.
        y_val = np.roll(y_val, -self.pred_len, axis=1)
        m_val = np.roll(m_val, -self.pred_len, axis=1)
        best, patience = float("inf"), self.patience
        logs = []
        rng = np.random.default_rng(self.seed)
        with tempfile.TemporaryDirectory(prefix="tfts-windowed-") as checkpoint_dir:
            ckpt = f"{checkpoint_dir}/best.weights.h5"
            for ep in range(self.epochs):
                x, y, m = sampled_windows(
                    train_histories, rng, self.seq_len, self.pred_len, num_features=self.num_features
                )
                hist = self.model.fit(x, y, sample_weight=m, batch_size=self.batch_size, epochs=1, verbose=0)
                prediction = self.model.predict(x_val, batch_size=self.batch_size, verbose=0)
                element_loss = _smape_loss_fn(y_val, prediction).numpy()
                val = float(np.sum(element_loss * m_val) / max(1.0, np.sum(m_val)))
                logs.append({"epoch": ep + 1, "train": float(hist.history["loss"][-1]), "val": val})
                if verbose:
                    print(
                        f"  [tfts] epoch {ep + 1}/{self.epochs} " f"train={hist.history['loss'][-1]:.3f} val={val:.3f}",
                        flush=True,
                    )
                if val < best:
                    best, patience = val, self.patience
                    self.model.save_weights(ckpt)
                else:
                    patience -= 1
                    if patience <= 0:
                        if verbose:
                            print(f"  [tfts] early stop at epoch {ep + 1}", flush=True)
                        break
            self.model.load_weights(ckpt)
        return logs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forecast ``[N, seq_len, features]`` -> ``[N, pred_len, features]``."""
        out = self.model(tf.constant(x, dtype=tf.float32), training=False)
        return tf.cast(out, tf.float32).numpy()

    def evaluate(self, histories: Sequence[np.ndarray], targets: np.ndarray) -> dict:
        """Score the held-out final window of every series (SMAPE/MAE/MSE)."""
        x_final, _ = final_windows(histories, self.seq_len, self.num_features)
        pred = self.predict(x_final)
        true = np.asarray(targets)
        if true.ndim == 2 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        return dict(
            smape=smape_score(pred, true),
            mae=float(np.mean(np.abs(pred - true))),
            mse=float(np.mean((pred - true) ** 2)),
            prediction=pred,
            target=true,
        )
