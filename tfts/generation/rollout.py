"""High-level forecast rollout strategies."""

from abc import ABC, abstractmethod
from dataclasses import fields, replace

import tensorflow as tf

from tfts.contracts import ForecastMode

from .decoding import decode
from .engine import TimeAxisEngine
from .outputs import ForecastGenerationOutput
from .processors import resolve_forecast_processors
from .samplers import StepOutput, resolve_value_sampler
from .steps import StepDecoder


def _future_step(values, step):
    """Return a validated future timestep without extrapolating covariates."""
    if values is None:
        return None
    tf.debugging.assert_less(step, tf.shape(values)[1], message="future covariates are too short")
    return tf.expand_dims(tf.gather(values, step, axis=1), axis=1)


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
        probabilistic=model.generation_probabilistic,
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
    if batch.structure is not None:
        per_sample, shared = batch.structure.split_tensor_dict()
        if per_sample:
            tensors = {**shared, **{name: tf.repeat(value, repeats, axis=0) for name, value in per_sample.items()}}
            values["structure"] = type(batch.structure).from_tensor_dict(tensors)
    return replace(batch, **values)


def _aggregate_trajectories(trajectories, aggregation):
    if aggregation == "mean":
        return tf.reduce_mean(trajectories, axis=1)
    if aggregation == "median":
        ordered = tf.sort(trajectories, axis=1)
        count = tf.shape(trajectories)[1]
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
    def run(self, model, batch, config, *, horizon, sampler=None, processors=None, stopping_criteria=None):
        raise NotImplementedError


class SampleAggregator(RolloutStrategy):
    """Apply sample breadth and aggregation uniformly around any rollout."""

    def __init__(self, inner):
        self.inner = inner

    def run(self, model, batch, config, *, horizon, sampler=None, processors=None, stopping_criteria=None):
        active_sampler = _resolve_sampler(model, sampler, config)
        count = config.num_samples if active_sampler.stochastic else 1
        source = _repeat_batch(batch, count) if count > 1 else batch
        output = self.inner.run(
            model,
            source,
            config,
            horizon=horizon,
            sampler=active_sampler,
            processors=processors,
            stopping_criteria=stopping_criteria,
        )

        def trajectories(value):
            return tf.reshape(value, tf.concat([[batch.batch_size, count], tf.shape(value)[1:]], axis=0))

        samples = trajectories(output.predictions)
        parameters = (
            None
            if output.distribution_params is None
            else {
                name: trajectories(value) if count > 1 else value for name, value in output.distribution_params.items()
            }
        )
        return ForecastGenerationOutput(
            predictions=_aggregate_trajectories(samples, config.aggregation),
            samples=samples if config.return_samples else None,
            distribution_params=parameters,
            quantile_values=(
                None
                if output.quantile_values is None
                else (trajectories(output.quantile_values) if count > 1 else output.quantile_values)
            ),
            values_processed=any(p.stage == "values" for p in resolve_forecast_processors(processors)),
        )


class DirectRollout(RolloutStrategy):
    def run(self, model, batch, config, *, horizon, sampler=None, processors=None, stopping_criteria=None):
        active_sampler = _resolve_sampler(model, sampler, config)
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        output = model.forward(batch, training=False)
        tf.debugging.assert_greater_equal(
            tf.shape(output.predictions)[1],
            horizon,
            message="Direct forecast is shorter than prediction_length; use strategy='recursive'",
        )

        def step_fn(previous, state, offset):
            return StepOutput(
                output.predictions[:, offset : offset + 1, ...],
                parameters=tf.nest.map_structure(
                    lambda value: value[:, offset : offset + 1, ...], output.distribution_params or {}
                ),
                distribution=model.output_distribution,
                quantile_values=(
                    None if output.quantile_values is None else output.quantile_values[:, offset : offset + 1, ...]
                ),
            )

        result = TimeAxisEngine(active_sampler, processors=processors, stopping_criteria=stopping_criteria).run(
            step_fn,
            batch.past_values[:, -1:, ...],
            None,
            horizon,
            past_values=batch.past_values,
            seed_for_step=lambda offset: (
                None if config.seed is None else tf.stack([tf.cast(config.seed, tf.int32), offset])
            ),
        )
        return ForecastGenerationOutput(
            predictions=result.values,
            distribution_params=result.distribution_params,
            quantile_values=result.quantile_values,
        )


class RecursiveRollout(RolloutStrategy):
    def run(self, model, batch, config, *, horizon, sampler=None, processors=None, stopping_criteria=None):
        active_sampler = _resolve_sampler(model, sampler, config)
        _validate_future_horizon(batch, horizon)
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        source = batch

        def next_input(current, value, *, offset):
            current = replace(source, **current)
            return _shift_recursive_batch(current, source, value, offset).as_tensor_dict(include_structure=False)

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
                prediction=output.predictions[:, :1, ...],
                distribution=model.output_distribution,
                quantile_values=None if output.quantile_values is None else output.quantile_values[:, :1, ...],
                parameters=_one_step_parameters(output.distribution_params),
                state=model_state,
            )

        def seed_for_step(step):
            return None if config.seed is None else tf.stack([tf.cast(config.seed, tf.int32), tf.cast(step, tf.int32)])

        rollout = TimeAxisEngine(active_sampler, processors=processors, stopping_criteria=stopping_criteria).run(
            StepDecoder(step_fn, next_input),
            replace(source, future_values=None, future_observed_mask=None, labels=None).as_tensor_dict(
                include_structure=False
            ),
            None,
            horizon,
            past_values=batch.past_values,
            seed_for_step=seed_for_step,
        )
        return ForecastGenerationOutput(
            predictions=rollout.values,
            distribution_params=rollout.distribution_params,
            quantile_values=rollout.quantile_values,
        )


class AutoregressiveRollout(RolloutStrategy):
    def run(self, model, batch, config, *, horizon, sampler=None, processors=None, stopping_criteria=None):
        if ForecastMode.AUTOREGRESSIVE not in model.capabilities.forecast_modes:
            raise ValueError("This backbone does not declare incremental decoding")
        active_sampler = _resolve_sampler(model, sampler, config)
        batch = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        model_batch, restore = model.prepare_backbone_batch(batch)
        output = decode(
            model.backbone,
            model_batch,
            horizon,
            sampler=active_sampler,
            processors=processors,
            seed=config.seed,
            stopping_criteria=stopping_criteria,
        )
        return ForecastGenerationOutput(
            predictions=restore(output.values),
            distribution_params=None if output.distribution_params is None else restore(output.distribution_params),
            quantile_values=None if output.quantile_values is None else restore(output.quantile_values),
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
