"""Public generation entry point."""

from dataclasses import replace

import tensorflow as tf

from tfts.contracts import TimeSeriesBatch
from tfts.data.sequence_utils import pad_sequences

from .configuration import ForecastGenerationConfig
from .rollout import SampleAggregator, resolve_rollout_strategy


def prepare_generation_batch(
    sequences,
    *,
    sequence_length=None,
    padding_side="left",
    pad_value=0.0,
    **fields,
):
    """Build a canonical generation batch from variable-length histories.

    The returned ``padding_mask`` uses TFTS's convention: ``True`` marks a
    valid timestep. Additional :class:`TimeSeriesBatch` fields can be supplied
    through ``fields``.
    """
    if "past_values" in fields:
        raise TypeError("pass histories as sequences, not as the past_values field")
    values, padding_mask = pad_sequences(
        sequences,
        sequence_length=sequence_length,
        padding_side=padding_side,
        pad_value=pad_value,
        return_padding_mask=True,
    )
    if not values.dtype.kind == "f":
        values = values.astype("float32")
    fields = dict(fields)
    fields.setdefault("padding_mask", padding_mask)
    return TimeSeriesBatch(past_values=values, **fields)


def generate(
    model,
    inputs,
    generation_config=None,
    *,
    strategy=None,
    sampler=None,
    processors=None,
    stopping_criteria=None,
    **kwargs,
):
    config = ForecastGenerationConfig.from_args(generation_config, **kwargs)
    rollout = resolve_rollout_strategy(
        model,
        strategy if strategy is not None else config.strategy,
        prediction_length=config.prediction_length,
    )
    batch = TimeSeriesBatch.from_inputs(inputs)
    batch.validate_for("forecasting")
    if not batch.past_values.dtype.is_floating:
        batch = replace(batch, past_values=tf.cast(batch.past_values, tf.float32))
    if batch.padding_mask is not None:
        supports_padding = model.capabilities.supports_variable_length
        if not supports_padding:
            tf.debugging.assert_equal(
                tf.reduce_all(batch.padding_mask),
                True,
                message=(
                    "This forecasting backbone does not support padded histories. "
                    "Use equal-length histories or a backbone that declares supports_variable_length=True."
                ),
            )
    horizon = config.prediction_length if config.prediction_length is not None else model.task_config.prediction_length
    return SampleAggregator(rollout).run(
        model,
        batch,
        config,
        horizon=horizon,
        sampler=sampler,
        processors=processors,
        stopping_criteria=stopping_criteria,
    )
