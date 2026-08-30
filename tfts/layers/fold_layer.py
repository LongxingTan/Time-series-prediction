"""Weight-free spatial reshape layers and batch transformations."""

import tensorflow as tf


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
