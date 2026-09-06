"""Training compatibility helpers using the shared decoder and feedback policy."""

from __future__ import annotations

from typing import Mapping, Union

import tensorflow as tf

from .schedules import teacher_forcing_decay

__all__ = [
    "scheduled_sampling_decode",
    "teacher_forcing_decay",
]


def scheduled_sampling_decode(
    model,
    x: tf.Tensor,
    static: tf.Tensor,
    y_true: tf.Tensor,
    teacher_prob,
    stochastic: bool = True,
    seed: int = 0,
    distribution_output=None,
) -> Union[Mapping[str, tf.Tensor], tf.Tensor]:
    """Sequential decoder pass mixing teacher-forced targets with the model's own samples.

    Compatibility adapter. New callers should use ``generation.decode`` with a
    ``TimeSeriesBatch`` and explicit ``teacher_probability``.

    Parameters
    ----------
    model : an autoregressive model exposing ``initialize_decode`` and
        ``decode_step`` (e.g. ``tfts`` DeepAR), or its task wrapper.
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
    A mapping of distribution parameter names to ``(B, pred_len, F)`` tensors,
    or predictions for a deterministic decoder.
    Gradients flow to ``model``'s variables.
    """
    from tfts.contracts import TimeSeriesBatch
    from tfts.generation.decoding import decode

    model = getattr(model, "backbone", model)
    if distribution_output is not None and distribution_output is not getattr(model, "output_distribution", None):
        raise ValueError("distribution_output must be the model's shared output head")
    batch = TimeSeriesBatch(past_values=x, static_categorical_features=static, future_values=y_true)
    result = decode(
        model,
        batch,
        tf.shape(y_true)[1],
        training=True,
        teacher_probability=teacher_prob,
        sampler="sample" if stochastic else "point",
        seed=seed,
    )
    return result.distribution_params if result.distribution_params is not None else result.predictions
