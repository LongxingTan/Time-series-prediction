"""The only chunk assembly loop used by generation."""

from __future__ import annotations

import tensorflow as tf

from .chunk import Chunk
from .processors import StepProcessorList
from .stopping import StoppingCriteriaList
from .trajectory import Trajectory


def _time_first(value):
    return tf.transpose(value, tf.concat([[1, 0], tf.range(2, tf.rank(value))], axis=0))


def _batch_first(value):
    return tf.transpose(value, tf.concat([[1, 0], tf.range(2, tf.rank(value))], axis=0))


def _write(buffer, value, offset, horizon):
    width = tf.minimum(tf.shape(value)[1], horizon - offset)
    indices = tf.range(offset, offset + width)[:, None]
    return tf.tensor_scatter_nd_update(buffer, indices, _time_first(value[:, :width, ...]))


def _seed(seed, offset):
    base = 0 if seed is None else seed
    return tf.stack([tf.cast(base, tf.int32), tf.cast(offset, tf.int32)])


def run(decoder, batch, *, horizon, processors, stopping=(), training=False, seed=None):
    """Decode chunks, process them in list order, and assemble one trajectory."""

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive Python int")
    processors = processors if isinstance(processors, StepProcessorList) else StepProcessorList(processors)
    stopping = stopping if isinstance(stopping, StoppingCriteriaList) else StoppingCriteriaList(stopping)
    if not isinstance(decoder.output_chunk_length, int) or decoder.output_chunk_length < 1:
        raise ValueError("output_chunk_length must be a positive Python int")

    state = decoder.start(batch, horizon=horizon, training=training)
    first = decoder.step(state, offset=tf.constant(0, tf.int32), training=training)
    tf.debugging.assert_greater_equal(
        tf.shape(first.prediction)[1],
        min(horizon, decoder.output_chunk_length),
        message="decoder emitted a short chunk",
    )
    buffer = tf.zeros(
        tf.concat([[horizon, tf.shape(first.prediction)[0]], tf.shape(first.prediction)[2:]], axis=0),
        first.prediction.dtype,
    )

    parameter_buffers = None
    quantile_buffer = None
    offset = 0
    chunk = first
    stopped = False
    while offset < horizon:
        generated = _batch_first(buffer[:offset]) if processors.needs_history else chunk.prediction[:, :0, ...]
        chunk = chunk.replace(
            offset=tf.constant(offset, tf.int32),
            generated=generated,
            seed=_seed(seed, offset),
        )
        chunk = processors(chunk)
        tf.debugging.assert_equal(
            tf.shape(chunk.value),
            tf.shape(chunk.prediction),
            message="processors must preserve the emitted chunk shape",
        )
        width = min(decoder.output_chunk_length, horizon - offset)
        value = chunk.value[:, :width, ...]
        buffer = _write(buffer, value, offset, horizon)

        if chunk.parameters is not None:
            if parameter_buffers is None:
                parameter_buffers = {
                    name: tf.zeros(
                        tf.concat([[horizon, tf.shape(item)[0]], tf.shape(item)[2:]], axis=0),
                        item.dtype,
                    )
                    for name, item in chunk.parameters.items()
                }
            parameter_buffers = {
                name: _write(parameter_buffers[name], item[:, :width, ...], offset, horizon)
                for name, item in chunk.parameters.items()
            }
        if chunk.quantiles is not None:
            if quantile_buffer is None:
                quantile_buffer = tf.zeros(
                    tf.concat([[horizon, tf.shape(chunk.quantiles)[0]], tf.shape(chunk.quantiles)[2:]], axis=0),
                    chunk.quantiles.dtype,
                )
            quantile_buffer = _write(quantile_buffer, chunk.quantiles[:, :width, ...], offset, horizon)

        stopped_value = stopping(chunk)
        if tf.executing_eagerly() and bool(stopped_value.numpy()):
            stopped = True
        offset += width
        if stopped or offset >= horizon:
            break
        state = decoder.feed(state, chunk.feedback[:, :width, ...], offset=tf.constant(offset - width, tf.int32))
        chunk = decoder.step(state, offset=tf.constant(offset, tf.int32), training=training)

    predictions = _batch_first(buffer)[:, :offset, ...]
    parameters = None
    if parameter_buffers is not None:
        parameters = {name: _batch_first(value)[:, :offset, ...] for name, value in parameter_buffers.items()}
    quantile_values = None if quantile_buffer is None else _batch_first(quantile_buffer)[:, :offset, ...]
    return Trajectory(
        predictions=predictions,
        distribution_params=parameters,
        quantile_values=quantile_values,
        distribution=getattr(decoder.model, "output_distribution", None),
    )


__all__ = ["run"]
