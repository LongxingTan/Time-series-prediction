"""Exposure-bias noise injection for autoregressive (DeepAR-style) training.

A DeepAR / probabilistic RNN trained purely by teacher forcing learns the correct
one-step conditional but can drift under its own sampled feed-back at inference
(autoregressive "exposure bias"). Injecting Gaussian noise into the decoder's
lagged-target input during training makes the model robust to that feed-back -- the
single biggest lever for closing the train/infer gap in this repo's DeepAR parity
work (TensorFlow generative MAE 0.857 -> 0.265 on the synthetic AR task).

The injected noise is *position-ramped* (``std(t)`` grows toward the end of the
horizon), which matches ancestral sampling: feed-back error accumulates with the
number of decoded steps. Position 0 is the clean last-encoder seed target and is
left untouched.

Both a TensorFlow-graph path (safe inside ``@tf.function`` training steps) and a
NumPy path (for eager/numpy batch loops) are provided, plus a linear annealing
schedule for noise_std across epochs.

Reference: Salinas et al. 2020, "DeepAR: Probabilistic Forecasting with
Autoregressive Recurrent Networks" (https://arxiv.org/abs/1704.04110).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf

__all__ = [
    "position_ramp",
    "add_exposure_bias_noise",
    "add_exposure_bias_noise_np",
    "annealed_noise_std",
]


def position_ramp(horizon: int, ramp_max: float = 2.5) -> np.ndarray:
    """Per-position noise amplification ``ramp[t] = 1 + (ramp_max-1)*t/(horizon-1)``.

    ``ramp[0]`` is 1.0 (position 0 is the clean seed and is zeroed externally by the
    ``keep_position0`` flag); ``ramp[horizon-1] == ramp_max``.
    """
    horizon = int(horizon)
    t = np.arange(horizon, dtype=np.float32)
    denom = float(horizon - 1) if horizon > 1 else 1.0
    return 1.0 + (float(ramp_max) - 1.0) * t / denom


def add_exposure_bias_noise(
    decoder_feature,
    noise_std: float = 0.1,
    ramp_max: float = 2.5,
    keep_position0: bool = True,
):
    """Add position-ramped Gaussian noise to the decoder feed-back positions.

    TensorFlow-graph friendly: the internal ops are ``tf.random.normal`` / ``tf.where``
    so it can run inside a ``@tf.function`` training step. ``decoder_feature`` may be a
    ``tf.Tensor`` or NumPy array with shape ``(B, P, F)``.

    Parameters
    ----------
    decoder_feature : (B, P, F) lagged-target decoder input (teacher-forced).
    noise_std : scalar std of the injected noise (scaled by the position ramp).
    ramp_max : amplification of ``noise_std`` at the last decoder position.
    keep_position0 : if True leave position 0 (the clean last-encoder seed target) unperturbed.

    Returns
    -------
    A tensor of the same shape/dtype as ``decoder_feature`` with noise added.
    """
    df = tf.convert_to_tensor(decoder_feature)
    P = tf.shape(df)[1]
    t = tf.cast(tf.range(P), df.dtype)  # (P,)
    denom = tf.cast(tf.maximum(P - 1, 1), df.dtype)
    ramp = 1.0 + (tf.cast(ramp_max, df.dtype) - 1.0) * t / denom  # (P,)
    if keep_position0:
        mask = tf.where(tf.equal(t, tf.cast(0.0, df.dtype)), tf.zeros_like(ramp), ramp)
    else:
        mask = ramp
    noise = tf.random.normal(tf.shape(df), mean=0.0, stddev=1.0, dtype=df.dtype) * mask[None, :, None]
    return df + tf.cast(noise_std, df.dtype) * noise


def add_exposure_bias_noise_np(
    decoder_feature: np.ndarray,
    noise_std: float = 0.1,
    ramp_max: float = 2.5,
    keep_position0: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """NumPy path of :func:`add_exposure_bias_noise` (for eager/numpy batch loops).

    Uses ``rng.normal`` directly (optionally seeded) and returns a ``float32`` array,
    e.g. ``db = add_exposure_bias_noise_np(db, noise_std, rng=noise_rng)`` inside an
    epoch loop before feeding the step.
    """
    df = np.asarray(decoder_feature, dtype=np.float32)
    P = df.shape[1]
    mask = position_ramp(P, ramp_max).copy()
    if keep_position0:
        mask[0] = 0.0
    rng = rng if rng is not None else np.random.default_rng()
    noise = rng.normal(0.0, 1.0, size=df.shape).astype(np.float32) * mask[None, :, None]
    return df + float(noise_std) * noise


def annealed_noise_std(
    epoch: int,
    warmup_epochs: int = 3,
    total_epochs: int = 40,
    start: float = 0.05,
    end: float = 0.5,
) -> float:
    """Linear anneal of ``noise_std`` from ``start`` (during warmup) to ``end``.

    Epochs 1..``warmup_epochs`` stay at ``start`` (the model first learns the clean
    one-step conditional); afterwards the injected noise grows linearly so that the
    final epoch noise is ``end``. This matches the schedule that produced the best
    DeepAR generative result in this repo.
    """
    epoch = max(1, int(epoch))
    if epoch <= int(warmup_epochs):
        return float(start)
    frac = min(1.0, max(0.0, (epoch - int(warmup_epochs)) / max(1, int(total_epochs) - int(warmup_epochs))))
    return float(start) + (float(end) - float(start)) * frac
