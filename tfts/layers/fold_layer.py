"""Weight-free spatial reshape layers and batch transformations."""

from dataclasses import fields
from typing import Callable

import tensorflow as tf

from tfts.contracts.batch import TimeSeriesBatch


def _fold_spatial_to_batch(inputs):
    rank = inputs.shape.rank
    if rank == 3:
        return inputs
    if rank is None:
        raise ValueError("FoldSpatialToBatch requires a statically known rank")
    shape = tf.shape(inputs)
    moved = tf.transpose(inputs, [0] + list(range(2, rank - 1)) + [1, rank - 1])
    return tf.reshape(moved, [-1, shape[1], shape[-1]])


def _unfold_batch_to_spatial(inputs, spatial_shape, batch_size=None):
    if not spatial_shape:
        return inputs
    shape = tf.shape(inputs)
    set_size = 1
    for dimension in spatial_shape:
        set_size *= dimension
    if batch_size is None:
        batch_size = shape[0] // set_size
    target = tf.concat([tf.reshape(batch_size, [1]), tf.constant(spatial_shape), shape[1:]], axis=0)
    restored = tf.reshape(inputs, target)
    spatial_rank = len(spatial_shape)
    rank = inputs.shape.rank
    if rank is None:
        raise ValueError("UnfoldBatchToSpatial requires a statically known rank")
    return tf.transpose(
        restored,
        [0, spatial_rank + 1] + list(range(1, spatial_rank + 1)) + list(range(spatial_rank + 2, spatial_rank + rank)),
    )


@tf.keras.utils.register_keras_serializable(package="tfts")
class FoldSpatialToBatch(tf.keras.layers.Layer):
    """Move spatial axes into the batch axis."""

    def call(self, inputs):
        return _fold_spatial_to_batch(inputs)


@tf.keras.utils.register_keras_serializable(package="tfts")
class UnfoldBatchToSpatial(tf.keras.layers.Layer):
    """Restore a folded tensor to ``[B,T,*S,C]``."""

    def __init__(self, spatial_shape, **kwargs):
        super().__init__(**kwargs)
        self.spatial_shape = tuple(int(dimension) for dimension in spatial_shape)

    def call(self, inputs, batch_size=None):
        return _unfold_batch_to_spatial(inputs, self.spatial_shape, batch_size=batch_size)

    def get_config(self):
        config = super().get_config()
        config["spatial_shape"] = self.spatial_shape
        return config


class SpatialBatchTransform:
    """Apply a spatial fallback consistently to every aligned batch field."""

    _TEMPORAL = {
        "past_values",
        "future_values",
        "past_time_features",
        "future_time_features",
        "past_categorical_features",
        "future_categorical_features",
        "past_observed_mask",
        "future_observed_mask",
    }
    _STATIC = {"static_real_features", "static_categorical_features"}

    def __init__(self, strategy: str):
        if strategy != "per_node":
            raise ValueError("spatial_strategy must be 'per_node'")
        self.strategy = strategy

    def apply(self, batch: TimeSeriesBatch):
        spatial_rank = len(batch.spatial_shape)
        if not batch.spatial_shape:
            raise ValueError("per_node requires rank-4 set or rank-5 grid values")
        set_size = 1
        for dimension in batch.spatial_shape:
            set_size *= dimension
        values = {}
        for field in fields(batch):
            name, value = field.name, getattr(batch, field.name)
            if name == "structure" or value is None:
                continue
            if not tf.is_tensor(value):
                values[name] = value
            elif name in self._TEMPORAL:
                values[name] = self._temporal(value, spatial_rank, set_size)
            elif name in self._STATIC:
                values[name] = self._static(value, spatial_rank, set_size)
            elif name in {"padding_mask", "labels"}:
                values[name] = tf.repeat(value, set_size, axis=0)
            else:
                values[name] = value
        return TimeSeriesBatch(**values), self._restore(batch)

    def _temporal(self, value, spatial_rank, set_size):
        if value.shape.rank == 3:
            return tf.repeat(value, set_size, axis=0)
        return _fold_spatial_to_batch(value)

    def _static(self, value, spatial_rank, set_size):
        if value.shape.rank == 2:
            return tf.repeat(value, set_size, axis=0)
        shape = tf.shape(value)
        return tf.reshape(value, [-1, shape[-1]])

    def _restore(self, original: TimeSeriesBatch) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda value: (
            None
            if value is None
            else _unfold_batch_to_spatial(value, original.spatial_shape, batch_size=original.batch_size)
        )
