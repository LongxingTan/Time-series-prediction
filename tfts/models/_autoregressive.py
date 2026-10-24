"""Shared contracts and input boundary for native autoregressive backbones."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import tensorflow as tf

from tfts.batch import TimeSeriesBatch
from tfts.models.base import BackboneCapabilities, BackboneOutput, BaseModel, ModelInputSpec

AUTOREGRESSIVE_CAPABILITIES = BackboneCapabilities(
    input_spec=ModelInputSpec(accepted_roles=frozenset({"observed_past", "known_future"})),
)


@dataclass(frozen=True)
class DecodeSession:
    """Transient backbone-owned context, state, and initial feedback."""

    context: Any
    state: Any
    previous: tf.Tensor


@dataclass(frozen=True)
class DecodeStepOutput:
    """One prediction shaped [B, 1, ..., target_dim] and updated decoder state."""

    prediction: tf.Tensor
    state: Any = None
    distribution: Optional[Any] = None
    parameters: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None


def decoder_features(batch, horizon):
    """Future covariates, or a relative forecast-position feature."""
    features = batch.future_time_features
    if features is None:
        features = tf.tile(tf.range(horizon)[None, :, None], [batch.batch_size, 1, 1])
    tf.debugging.assert_greater_equal(tf.shape(features)[1], horizon, message="future covariates are too short")
    return tf.cast(features[:, :horizon, :], batch.past_values.dtype)


def encoder_features(batch):
    values = batch.past_values
    if batch.past_time_features is not None:
        values = tf.concat([values, tf.cast(batch.past_time_features, values.dtype)], axis=-1)
    return values


class AutoregressiveModel(BaseModel):
    """Base for backbones with a native incremental decoder."""

    def __call__(self, batch, *args, **kwargs):
        # Expose tensor shapes to Keras before it builds the backbone. A batch
        # dataclass itself has no shape and cannot serve as a Keras build spec.
        if isinstance(batch, TimeSeriesBatch):
            batch = batch.as_tensor_dict()
        return super().__call__(batch, *args, **kwargs)

    def next_input(self, value, context, *, offset):
        return value

    def decoder_seed(self, batch):
        """Seed feedback with target channels, retaining all inputs for encoding.

        Legacy arrays can contain additional observed channels after the targets.
        The seed must have the same width as subsequent decoder predictions.
        """
        target_dim = self.config.target_dim
        tf.debugging.assert_greater_equal(
            tf.shape(batch.past_values)[-1], target_dim, message="history has fewer channels than target_dim"
        )
        return batch.past_values[:, -1:, :target_dim]

    def call(self, batch: TimeSeriesBatch, training=None):
        from tfts.generation.decoders import NativeDecoder
        from tfts.generation.loop import run
        from tfts.generation.processors import Mean, StepProcessorList

        batch = TimeSeriesBatch.from_inputs(batch)

        class _TaskView:
            head = None
            backbone = self
            output_distribution = getattr(self, "output_distribution", None)

            @staticmethod
            def prepare_backbone_batch(value):
                return value, lambda tensor: tensor

        trajectory = run(
            NativeDecoder(_TaskView()),
            batch,
            horizon=self.predict_sequence_length,
            processors=StepProcessorList([Mean(getattr(self, "output_distribution", None))]),
            training=training,
        )
        return BackboneOutput(
            native_forecast=trajectory.predictions,
            distribution_params=trajectory.distribution_params,
        )
