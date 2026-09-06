"""High-level forecast rollout strategies."""

from abc import ABC, abstractmethod
from dataclasses import fields, replace

import tensorflow as tf

from tfts.contracts import ForecastMode

from .engine import GenerationEngine
from .outputs import ForecastGenerationOutput
from .processors import resolve_forecast_processors
from .samplers import DistributionSampler, SamplingResult, StepOutput, resolve_value_sampler


def _process_values(values, processors, horizon):
    if not processors:
        return values
    history = None
    selected = []
    for step in range(int(horizon)):
        result = SamplingResult(values[:, step : step + 1, :])
        result = processors(history, result, step=step, step_output=StepOutput(result.value))
        selected.append(result.value)
        history = tf.concat(selected, axis=1)
    return tf.concat(selected, axis=1)


def _future_step(values, step):
    """Return one future timestep, clamping when the covariate horizon is short."""
    if values is None or values.shape[1] == 0:
        return None
    if values.shape[1] is not None:
        index = tf.minimum(tf.cast(step, tf.int32), int(values.shape[1]) - 1)
    else:
        index = tf.minimum(tf.cast(step, tf.int32), tf.shape(values)[1] - 1)
    return tf.expand_dims(tf.gather(values, index, axis=1), axis=1)


def _feature_names(batch, name):
    metadata = batch.metadata or {}
    return tuple(metadata.get("feature_names", {}).get(name, ()))


def _shift_past_field(current, future, step, context_length, *, past_names=(), future_names=(), name="features"):
    if current is None:
        next_row = _future_step(future, step)
        return None if next_row is None else tf.repeat(next_row, context_length, axis=1)
    future_row = _future_step(future, step)
    if future_row is None:
        next_row = current[:, -1:, ...]
    elif past_names or future_names:
        if len(past_names) != current.shape[-1] or len(future_names) != future.shape[-1]:
            raise ValueError(f"{name} metadata does not match its tensor width")
        unknown = set(future_names) - set(past_names)
        if unknown:
            raise ValueError(f"future {name} are absent from the past layout: {sorted(unknown)}")
        columns = []
        future_positions = {value: index for index, value in enumerate(future_names)}
        for index, feature_name in enumerate(past_names):
            if feature_name in future_positions:
                columns.append(future_row[..., future_positions[feature_name] : future_positions[feature_name] + 1])
            else:
                columns.append(current[:, -1:, ..., index : index + 1])
        next_row = tf.concat(columns, axis=-1)
    else:
        tf.debugging.assert_equal(
            tf.shape(current)[-1],
            tf.shape(future_row)[-1],
            message=f"{name} layouts differ. Provide feature_names metadata for explicit mapping",
        )
        next_row = future_row
    return tf.concat([current[:, 1:, ...], next_row], axis=1)


def _shift_recursive_batch(current, source, value, step):
    """Feed one recursive prediction back into a canonical batch.

    ``source`` is kept immutable so future-known covariates can be read at the
    original forecast step while ``current`` contains the rolling context.
    """
    tf.debugging.assert_equal(
        tf.shape(value)[-1],
        tf.shape(current.past_values)[-1],
        message="recursive predictions must match the target dimension",
    )
    next_values = tf.cast(value, current.past_values.dtype)
    next_observed_mask = None
    if current.past_observed_mask is not None:
        next_observed_mask = tf.zeros_like(current.past_observed_mask[:, -1:, ...])
    next_padding_mask = None
    if current.padding_mask is not None:
        next_padding_mask = tf.ones_like(current.padding_mask[:, -1:, ...])
    return replace(
        current,
        past_values=tf.concat([current.past_values[:, 1:, ...], next_values], axis=1),
        past_time_features=_shift_past_field(
            current.past_time_features,
            source.future_time_features,
            step,
            tf.shape(current.past_values)[1],
            past_names=_feature_names(source, "past_real"),
            future_names=_feature_names(source, "future_real"),
            name="real features",
        ),
        past_categorical_features=_shift_past_field(
            current.past_categorical_features,
            source.future_categorical_features,
            step,
            tf.shape(current.past_values)[1],
            past_names=_feature_names(source, "past_categorical"),
            future_names=_feature_names(source, "future_categorical"),
            name="categorical features",
        ),
        past_observed_mask=(
            None
            if current.past_observed_mask is None
            else tf.concat([current.past_observed_mask[:, 1:, ...], next_observed_mask], axis=1)
        ),
        padding_mask=(
            None
            if current.padding_mask is None
            else tf.concat([current.padding_mask[:, 1:, ...], next_padding_mask], axis=1)
        ),
        future_values=None,
        future_observed_mask=None,
        labels=None,
        future_time_features=(
            None if source.future_time_features is None else source.future_time_features[:, step + 1 :, ...]
        ),
        future_categorical_features=(
            None
            if source.future_categorical_features is None
            else source.future_categorical_features[:, step + 1 :, ...]
        ),
    )


def _resolve_sampler(model, sampler, config):
    return resolve_value_sampler(
        sampler if sampler is not None else config.sampler,
        probabilistic=getattr(model, "output_distribution", None) is not None,
    )


def _one_step_parameters(parameters):
    if parameters is None:
        return None
    return tf.nest.map_structure(lambda value: value[:, :1, ...], parameters)


def _repeat_batch(batch, repeats):
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = tf.repeat(value, repeats, axis=0) if tf.is_tensor(value) else value
    return type(batch)(**values)


def _aggregate_trajectories(trajectories, aggregation):
    if aggregation == "mean":
        return tf.reduce_mean(trajectories, axis=1)
    if aggregation == "median":
        ordered = tf.sort(trajectories, axis=1)
        count = trajectories.shape[1]
        lower = ordered[:, (count - 1) // 2, ...]
        upper = ordered[:, count // 2, ...]
        return (lower + upper) / 2.0
    return trajectories[:, 0, ...]


def _validate_future_horizon(batch, horizon):
    for name in ("future_time_features", "future_categorical_features"):
        value = getattr(batch, name)
        if value is not None:
            tf.debugging.assert_greater_equal(
                tf.shape(value)[1],
                horizon,
                message=f"{name} must cover the requested recursive prediction length",
            )


class RolloutStrategy(ABC):
    @abstractmethod
    def run(self, model, batch, config, sampler=None, processors=None):
        raise NotImplementedError


class DirectRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        active_sampler = _resolve_sampler(model, sampler, config)
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        output = model.forward(batch, training=False)
        values = output.predictions
        horizon = config.prediction_length or model.task_config.prediction_length
        tf.debugging.assert_greater_equal(
            tf.shape(values)[1],
            horizon,
            message="Direct forecast is shorter than prediction_length; use strategy='recursive'",
        )
        values = values[:, :horizon, :]
        parameters = tf.nest.map_structure(lambda value: value[:, :horizon, ...], output.distribution_params or {})
        sample_count = config.num_samples if isinstance(active_sampler, DistributionSampler) else 1
        selected = active_sampler.sample(
            StepOutput(
                tf.repeat(values, sample_count, axis=0),
                distribution=getattr(model, "output_distribution", None),
                parameters=tf.nest.map_structure(lambda value: tf.repeat(value, sample_count, axis=0), parameters),
            ),
            step=0,
            seed=config.seed,
        ).value
        tf.debugging.assert_equal(
            tf.shape(selected),
            tf.concat([[tf.shape(values)[0] * sample_count], tf.shape(values)[1:]], axis=0),
            message="sampler must preserve the forecast block shape",
        )
        selected = _process_values(selected, resolve_forecast_processors(processors), horizon)
        trajectories = tf.reshape(
            selected, tf.concat([[tf.shape(values)[0], sample_count], tf.shape(values)[1:]], axis=0)
        )
        return ForecastGenerationOutput(
            predictions=_aggregate_trajectories(trajectories, config.aggregation),
            samples=trajectories if config.return_samples else None,
            distribution_params=parameters or None,
            quantile_values=(None if output.quantile_values is None else output.quantile_values[:, :horizon, ...]),
        )


class RecursiveRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        horizon = config.prediction_length or model.task_config.prediction_length
        active_sampler = _resolve_sampler(model, sampler, config)
        sample_count = config.num_samples if isinstance(active_sampler, DistributionSampler) else 1
        _validate_future_horizon(batch, horizon)
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        source = _repeat_batch(batch, sample_count) if sample_count > 1 else batch

        def feedback(current, result, step, context):
            current = replace(context, **current)
            return _shift_recursive_batch(current, context, result.value, step).as_tensor_dict(include_structure=False)

        def step_fn(current, model_state, step):
            current_batch = replace(source, **current)
            # Recursive feedback preserves context width even though the shared
            # engine permits growing decoder state and shrinking future fields.
            current_batch = replace(
                current_batch, past_values=tf.ensure_shape(current_batch.past_values, source.past_values.shape)
            )
            output = model.forward(current_batch, training=False)
            tf.debugging.assert_greater_equal(
                tf.shape(output.predictions)[1],
                1,
                message="recursive model output must contain at least one timestep",
            )
            return StepOutput(
                prediction=output.predictions[:, :1, :],
                distribution=getattr(model, "output_distribution", None),
                parameters=_one_step_parameters(output.distribution_params),
                state=model_state,
            )

        def seed_for_step(step):
            return None if config.seed is None else tf.stack([tf.cast(config.seed, tf.int32), tf.cast(step, tf.int32)])

        rollout = GenerationEngine(active_sampler, feedback, processors).run(
            step_fn,
            replace(source, future_values=None, future_observed_mask=None, labels=None).as_tensor_dict(
                include_structure=False
            ),
            None,
            horizon,
            context=source,
            seed_for_step=seed_for_step,
        )
        batch_size = tf.shape(batch.past_values)[0]
        target_dim = tf.shape(rollout.values)[-1]
        trajectories = tf.reshape(rollout.values, [batch_size, sample_count, horizon, target_dim])
        predictions = _aggregate_trajectories(trajectories, config.aggregation)
        return ForecastGenerationOutput(
            predictions=predictions,
            samples=trajectories if config.return_samples else None,
        )


class AutoregressiveRollout(RolloutStrategy):
    def run(self, model, batch, config, sampler=None, processors=None):
        from .decoding import decode

        if not hasattr(model.backbone, "initialize_decode"):
            raise ValueError("This backbone does not implement incremental decoding")
        horizon = config.prediction_length or model.task_config.prediction_length
        active_sampler = _resolve_sampler(model.backbone, sampler, config)
        sample_count = config.num_samples if isinstance(active_sampler, DistributionSampler) else 1
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        model_batch, restore = model.prepare_backbone_batch(batch)
        source = _repeat_batch(model_batch, sample_count) if sample_count > 1 else model_batch
        output = decode(
            model.backbone, source, horizon, sampler=active_sampler, processors=processors, seed=config.seed
        )
        # Restore spatial axes for each trajectory before aggregating samples.
        trajectories = tf.reshape(output.values, [model_batch.batch_size, sample_count, horizon, -1])
        restored = tf.stack([restore(trajectories[:, index, ...]) for index in range(sample_count)], axis=1)
        return ForecastGenerationOutput(
            predictions=_aggregate_trajectories(restored, config.aggregation),
            samples=restored if config.return_samples else None,
        )


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
