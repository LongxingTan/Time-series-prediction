"""Public generation entry point."""

from tfts.contracts import TimeSeriesBatch

from .configuration import ForecastGenerationConfig
from .rollout import resolve_rollout_strategy


def generate(model, inputs, generation_config=None, *, strategy=None, sampler=None, processors=None, **kwargs):
    config = ForecastGenerationConfig.from_args(generation_config, **kwargs)
    rollout = resolve_rollout_strategy(
        model,
        strategy if strategy is not None else config.strategy,
        prediction_length=config.prediction_length,
    )
    batch = TimeSeriesBatch.from_inputs(inputs)
    batch.validate_for("forecasting")
    return rollout.run(model, batch, config, sampler=sampler, processors=processors)
