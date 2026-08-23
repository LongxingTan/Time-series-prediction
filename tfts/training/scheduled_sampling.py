"""Scheduled-sampling (curriculum / exposure-bias) training for autoregressive models.

Pure teacher forcing trains an autoregressive RNN on clean one-step conditionals, so at
inference it drifts under its own sampled feed-back (exposure bias). Scheduled sampling
-- the seq2seq recipe, see ``tfts/models/seq2seq.py`` ``DecodecV1.call`` -- mixes, at each
decoder step, the true lagged target with the model's *own* previous prediction: with
probability ``teacher_prob`` the teacher-forced value is fed back, otherwise the sampled
value is. Annealing ``teacher_prob`` from 1.0 down over training makes the model robust to
its own feed-back, which is exactly the distribution it sees at ancestral-sampling inference.

This is more faithful than the Gaussian noise-injection proxy (``tfts/training/exposure_bias.py``)
because the perturbed input is a *draw from the model's predicted Normal*, not independent
noise. It reuses the two generation hooks every autoregressive model already implements
(``initialize_generation_state`` + ``decode_step``), so it works for DeepAR and any similar
model without changing the model itself.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = [
    "scheduled_sampling_decode",
    "teacher_forcing_decay",
]


def teacher_forcing_decay(
    epoch: int,
    warmup_epochs: int = 3,
    total_epochs: int = 40,
    end_teacher: float = 0.2,
) -> float:
    """Anneal the teacher-forcing probability from 1.0 down to ``end_teacher``.

    Epochs 1..``warmup_epochs`` stay fully teacher-forced (1.0); after warmup the
    probability of feeding the true target decays linearly to ``end_teacher`` by the
    final epoch, so the model increasingly relies on its own sampled feed-back —
    mirroring the inference-time distribution. 1.0 = full teacher forcing, 0.0 = fully
    autoregressive (own samples only).
    """
    epoch = max(1, int(epoch))
    if epoch <= int(warmup_epochs):
        return 1.0
    frac = min(1.0, max(0.0, (epoch - int(warmup_epochs)) / max(1, int(total_epochs) - int(warmup_epochs))))
    return 1.0 - frac * (1.0 - float(end_teacher))


def scheduled_sampling_decode(
    model,
    x: tf.Tensor,
    static: tf.Tensor,
    y_true: tf.Tensor,
    teacher_prob,
    stochastic: bool = True,
    seed: int = 0,
) -> tuple:
    """Sequential decoder pass mixing teacher-forced targets with the model's own samples.

    Parameters
    ----------
    model : an autoregressive model exposing ``initialize_generation_state``,
        ``decode_step`` and ``predict_sequence_length`` (e.g. ``tfts`` DeepAR).
    x : (B, enc_len, F) normalized encoder window.
    static : (B, 1) static covariates (series id for the embedding).
    y_true : (B, pred_len, F) true target (used both for the NLL at each step and as the
        teacher-fed lagged input).
    teacher_prob : scalar in [0, 1]; probability of feeding the true target instead of the
        model's own sampled prediction at each decoder step. Anneal via
        :func:`teacher_forcing_decay`.
    stochastic : if True feed back ``loc + scale*noise`` (ancestral draw); if False feed
        back ``loc`` (greedy).
    seed : int base seed for the teacher-mask and sample draws, which are drawn from
        *stateless* RNG (``tf.random.stateless_*``) keyed by ``(seed, step)``. This makes
        training deterministic regardless of whether the loop is traced by ``@tf.function``
        or run eagerly (notebooks), so scheduled-sampling has no hidden run-to-run RNG
        variance.

    Returns
    -------
    (loc, scale): each (B, pred_len, F) predicted Normal parameters, ready for an NLL loss
    against ``y_true``. Gradients flow to ``model``'s variables.
    """
    if isinstance(teacher_prob, (int, float)):
        teacher_prob = tf.cast(teacher_prob, tf.float32)
    # resolve the model instance that actually implements the generation hooks
    # (an ``AutoModel``/wrapped model exposes them via ``core_model``).
    if not hasattr(model, "initialize_generation_state"):
        core = getattr(model, "core_model", None)
        if core is not None and hasattr(core, "initialize_generation_state"):
            model = core
    state = model.initialize_generation_state(x, static)
    cur = x[:, -1:, :]  # first decoder input = last encoder target (matches DeepAR)
    pred_len = int(getattr(model, "predict_sequence_length", None) or tf.shape(y_true)[1])
    B = tf.shape(x)[0]
    seed = int(seed)
    locs, scales = [], []
    for t in range(pred_len):
        params, state = model.decode_step(cur, static, state, training=True)
        loc, scale = params["loc"], params["scale"]
        locs.append(loc)
        scales.append(scale)
        if t + 1 < pred_len:
            # stateless, per-step-seeded draws: deterministic in eager AND under tf.function
            t_i = tf.cast(t, tf.int64)
            sample_seed = tf.stack([tf.constant(seed, tf.int64), 2 * t_i])
            teacher_seed = tf.stack([tf.constant(seed + 1, tf.int64), 2 * t_i + 1])
            if stochastic:
                noise = tf.random.stateless_normal(tf.shape(loc), seed=sample_seed, dtype=loc.dtype)
                fed = loc + scale * noise
            else:
                fed = loc
            u = tf.random.stateless_uniform([B, 1, 1], seed=teacher_seed, minval=0.0, maxval=1.0, dtype=loc.dtype)
            feed_teacher = tf.cast(u < teacher_prob, loc.dtype)
            cur = feed_teacher * y_true[:, t : t + 1, :] + (1.0 - feed_teacher) * fed
    return tf.concat(locs, axis=1), tf.concat(scales, axis=1)
