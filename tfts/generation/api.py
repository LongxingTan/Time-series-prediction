"""Public generation entry point."""

from dataclasses import fields, replace

import tensorflow as tf

from tfts.contracts import TimeSeriesBatch
from tfts.data.sequence_utils import pad_sequences
from tfts.registry import ComponentSpec, build, resolve

from .config import GenerationConfig
from .decoders import decoder_for
from .loop import run
from .processors import Mean, Median, Sample, Selector, StepProcessorList


def prepare_generation_batch(
    sequences,
    *,
    sequence_length=None,
    padding_side="left",
    pad_value=0.0,
    **fields,
):
    if "past_values" in fields:
        raise TypeError("pass histories as sequences, not as past_values")
    values, padding_mask = pad_sequences(
        sequences,
        sequence_length=sequence_length,
        padding_side=padding_side,
        pad_value=pad_value,
        return_padding_mask=True,
    )
    if values.dtype.kind != "f":
        values = values.astype("float32")
    fields.setdefault("padding_mask", padding_mask)
    return TimeSeriesBatch(past_values=values, **fields)


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


def _selector_for(model, config):
    distribution = getattr(model, "output_distribution", None)
    if distribution is not None and config.num_samples > 1:
        return Sample(distribution)
    if getattr(model.task_config, "head", None) == "quantile":
        return Median()
    return Mean(distribution)


def _processors(model, config, processors):
    if processors is None:
        specifications = config.processors
        if len(specifications) == 1 and specifications[0].name == "auto":
            return StepProcessorList([_selector_for(model, config)])
        processors = specifications
    if isinstance(processors, StepProcessorList):
        return processors
    if callable(processors) and not isinstance(processors, (list, tuple)):
        processors = [processors]
    resolved = []
    for component in processors:
        if isinstance(component, (str, ComponentSpec, dict)):
            spec = ComponentSpec.from_value(component)
            registration = resolve("processor", spec.name)
            kwargs = dict(spec.kwargs)
            if isinstance(registration.factory, type) and issubclass(registration.factory, Selector):
                kwargs.setdefault("distribution", getattr(model, "output_distribution", None))
                if spec.name == "quantile":
                    kwargs.setdefault("levels", getattr(model.task_config, "quantiles", None))
            component = registration.factory(**kwargs)
        resolved.append(component)
    return StepProcessorList(resolved)


def _reshape_samples(value, batch_size, count):
    return tf.reshape(value, tf.concat([[batch_size, count], tf.shape(value)[1:]], axis=0))


def generate(
    model,
    inputs,
    config=None,
    *,
    decoder=None,
    processors=None,
    trajectory=None,
    stopping=None,
    **overrides,
):
    config = GenerationConfig.from_value(config, **overrides)
    batch = TimeSeriesBatch.from_inputs(inputs)
    batch.validate_for("forecasting")
    if not batch.past_values.dtype.is_floating:
        batch = replace(batch, past_values=tf.cast(batch.past_values, tf.float32))
    horizon = config.horizon
    if horizon is None:
        horizon = int(model.output_chunk_length)
    active_processors = _processors(model, config, processors)
    count = config.num_samples if active_processors.stochastic else 1
    source = _repeat_batch(batch, count) if count > 1 else batch
    active_decoder = decoder or decoder_for(model, config.mode, horizon=horizon)
    result = run(
        active_decoder,
        source,
        horizon=horizon,
        processors=active_processors,
        stopping=stopping or (),
        seed=config.seed,
    )
    if result.quantile_values is not None:
        result = result.replace(quantiles=tuple(model.task_config.quantiles))
    if count > 1:
        samples = _reshape_samples(result.predictions, batch.batch_size, count)
        parameters = result.distribution_params
        if parameters is not None:
            parameters = {name: _reshape_samples(value, batch.batch_size, count) for name, value in parameters.items()}
        quantile_values = result.quantile_values
        if quantile_values is not None:
            quantile_values = _reshape_samples(quantile_values, batch.batch_size, count)
        result = result.replace(
            predictions=tf.reduce_mean(samples, axis=1),
            samples=samples,
            distribution_params=parameters,
            quantile_values=quantile_values,
        )

    transforms = config.trajectory if trajectory is None else trajectory
    scaler = (batch.metadata or {}).get("scaler")
    for component in transforms:
        if isinstance(component, (str, ComponentSpec, dict)):
            spec = ComponentSpec.from_value(component)
            kwargs = dict(spec.kwargs)
            if spec.name == "inverse_scale":
                kwargs.setdefault("scaler", scaler)
            component = resolve("transform", spec.name).factory(**kwargs)
        if getattr(component, "requires_fit", False) and not getattr(component, "is_fitted", False):
            raise RuntimeError(f"{type(component).__name__} must be fitted before use")
        result = component(result)
    return result


__all__ = ["generate", "prepare_generation_batch"]
