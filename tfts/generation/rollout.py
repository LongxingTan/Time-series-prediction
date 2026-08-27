"""High-level forecast rollout strategies."""

from abc import ABC, abstractmethod
from dataclasses import replace

import tensorflow as tf

from tfts.contracts import ForecastMode

from .engine import GenerationEngine
from .outputs import ForecastGenerationOutput
from .processors import resolve_forecast_processors
from .samplers import MeanSampler, SamplingResult, StepOutput, resolve_value_sampler


def _process_values(values, processors, horizon):
    history = None
    selected = []
    for step in range(int(horizon)):
        result = SamplingResult(values[:, step : step + 1, :])
        result = processors(history, result, step=step, step_output=StepOutput(result.value))
        selected.append(result.value)
        history = tf.concat(selected, axis=1)
    return tf.concat(selected, axis=1)


class RolloutStrategy(ABC):
    @abstractmethod
    def run(self, model, batch, config, sampler=None, processors=None):
        raise NotImplementedError


class DirectRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        output = model.forward(batch, training=False)
        values = output.predictions
        horizon = config.prediction_length or model.task_config.prediction_length
        tf.debugging.assert_greater_equal(
            tf.shape(values)[1],
            horizon,
            message="Direct forecast is shorter than prediction_length; use strategy='recursive'",
        )
        values = values[:, :horizon, :]
        values = _process_values(values, resolve_forecast_processors(processors), horizon)
        return ForecastGenerationOutput(
            predictions=values,
            distribution_params=output.distribution_params,
            quantile_values=output.quantile_values,
        )


class RecursiveRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        horizon = config.prediction_length or model.task_config.prediction_length
        active_processors = resolve_forecast_processors(processors)
        values = []
        current = batch
        for step in range(int(horizon)):
            output = model.forward(current, training=False)
            value = output.predictions[:, :1, :]
            result = SamplingResult(value)
            history = tf.concat(values, axis=1) if values else None
            result = active_processors(history, result, step=step, step_output=StepOutput(value))
            value = result.value
            values.append(value)
            next_row = current.past_values[:, -1:, :]
            target_dim = tf.shape(value)[-1]
            next_row = tf.concat([value, next_row[..., target_dim:]], axis=-1)
            next_past_features = current.past_time_features
            if batch.future_time_features is not None:
                feature_index = tf.minimum(step, tf.shape(batch.future_time_features)[1] - 1)
                next_feature = batch.future_time_features[:, feature_index : feature_index + 1, :]
                if next_past_features is None:
                    next_past_features = tf.repeat(next_feature, tf.shape(current.past_values)[1], axis=1)
                else:
                    next_past_features = tf.concat([next_past_features[:, 1:, :], next_feature], axis=1)
            current = replace(
                current,
                past_values=tf.concat([current.past_values[:, 1:, :], next_row], axis=1),
                past_time_features=next_past_features,
            )
        return ForecastGenerationOutput(predictions=tf.concat(values, axis=1))


class AutoregressiveRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        backbone = model.backbone
        for hook in ("initialize_generation_state", "decode_step", "output_distribution"):
            if not hasattr(backbone, hook):
                raise ValueError("%s lacks autoregressive hook %s" % (model.config.model_type, hook))
        horizon = config.prediction_length or model.task_config.prediction_length
        sample_count = config.num_samples if config.sampler in {"auto", "sample"} else 1
        active_sampler = resolve_value_sampler(sampler or config.sampler, probabilistic=True)
        if isinstance(active_sampler, MeanSampler):
            sample_count = 1
        x = tf.repeat(batch.past_values, sample_count, axis=0)
        static = batch.static_categorical_features
        if static is None:
            static = tf.zeros([batch.batch_size, 1], tf.int32)
        static = tf.repeat(static, sample_count, axis=0)
        state = backbone.initialize_generation_state(x, static)
        current = x[:, -1:, :]

        def feedback(_, result, **kwargs):
            return result.value

        def step_fn(previous, model_state, step):
            params, next_state = backbone.decode_step(previous, static, model_state, training=False)
            return StepOutput(
                prediction=backbone.output_distribution.mean(params),
                distribution=backbone.output_distribution,
                parameters=params,
                state=next_state,
            )

        def seed_for_step(step):
            return None if config.seed is None else tf.constant([config.seed, step], tf.int32)

        rollout = GenerationEngine(active_sampler, feedback, processors).run(
            step_fn, current, state, horizon, seed_for_step=seed_for_step
        )
        batch_size = tf.shape(batch.past_values)[0]
        target_dim = tf.shape(rollout.values)[-1]
        trajectories = tf.reshape(rollout.values, [batch_size, sample_count, horizon, target_dim])
        if config.aggregation == "mean":
            predictions = tf.reduce_mean(trajectories, axis=1)
        elif config.aggregation == "median":
            ordered = tf.sort(trajectories, axis=1)
            lower = ordered[:, (sample_count - 1) // 2, :, :]
            upper = ordered[:, sample_count // 2, :, :]
            predictions = (lower + upper) / 2.0
        else:
            predictions = trajectories[:, 0, :, :]
        samples = trajectories if config.return_samples else None
        return ForecastGenerationOutput(predictions=predictions, samples=samples)


def resolve_rollout_strategy(model, name, prediction_length=None):
    if isinstance(name, RolloutStrategy):
        return name
    if name == "auto":
        if ForecastMode.AUTOREGRESSIVE in model.capabilities.forecast_modes:
            name = "autoregressive"
        elif prediction_length is not None and prediction_length > model.task_config.prediction_length:
            name = "recursive"
        else:
            name = "direct"
    mapping = {"direct": DirectRollout, "recursive": RecursiveRollout, "autoregressive": AutoregressiveRollout}
    try:
        return mapping[name]()
    except KeyError as error:
        raise ValueError("Generation strategy %r is not implemented" % name) from error
