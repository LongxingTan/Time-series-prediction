"""Adapters between the canonical batch and architecture implementations."""

from __future__ import annotations

from dataclasses import fields
import inspect
from typing import Any, Callable, Dict

import tensorflow as tf

from tfts.contracts import BackboneOutput, OutputPort, TimeSeriesBatch
from tfts.layers.fold_layer import _fold_spatial_to_batch, _unfold_batch_to_spatial


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

    def apply(self, batch: TimeSeriesBatch):
        if not batch.spatial_shape:
            raise ValueError("per_node requires spatial values")
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
                values[name] = self._temporal(value, set_size)
            elif name in self._STATIC:
                values[name] = self._static(value, set_size)
            elif name in {"padding_mask", "labels"}:
                values[name] = tf.repeat(value, set_size, axis=0)
            else:
                values[name] = value
        return TimeSeriesBatch(**values), self._restore(batch)

    @staticmethod
    def _temporal(value, set_size):
        if value.shape.rank == 3:
            return tf.repeat(value, set_size, axis=0)
        return _fold_spatial_to_batch(value)

    @staticmethod
    def _static(value, set_size):
        if value.shape.rank == 2:
            return tf.repeat(value, set_size, axis=0)
        shape = tf.shape(value)
        return tf.reshape(value, [-1, shape[-1]])

    @staticmethod
    def _restore(original: TimeSeriesBatch) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda value: (
            None
            if value is None
            else _unfold_batch_to_spatial(value, original.spatial_shape, batch_size=original.batch_size)
        )


class SpatialAdapter:
    """Bidirectional coercion between a batch arrangement and one backbone."""

    def __init__(self, input_spec, model_type, strategy="raise"):
        if strategy not in {"raise", "per_node"}:
            raise ValueError("spatial_strategy must be 'raise' or 'per_node'")
        self.input_spec = input_spec
        self.model_type = model_type
        self.strategy = strategy
        self.transform = SpatialBatchTransform() if strategy == "per_node" else None

    def to_backbone(self, batch):
        from tfts.models.registry import check_batch_support

        if batch.arrangement == self.input_spec.arrangement:
            check_batch_support(self.model_type, batch, spec=self.input_spec)
            return batch, lambda value: value
        if self.transform is None:
            check_batch_support(self.model_type, batch, spec=self.input_spec)
            raise ValueError(f"{self.model_type!r} cannot adapt the {batch.arrangement.value} arrangement")
        transformed, restore = self.transform.apply(batch)
        check_batch_support(self.model_type, transformed, spec=self.input_spec)
        return transformed, restore

    def from_backbone(self, value, restore):
        if isinstance(value, dict):
            return {name: self.from_backbone(item, restore) for name, item in value.items()}
        return restore(value)


class BackboneAdapter:
    """Invoke one backbone without leaking its historical input/output shape."""

    def __init__(self, backbone, capabilities):
        self.backbone = backbone
        self.capabilities = capabilities
        self.model_type = getattr(backbone.config, "model_type", type(backbone).__name__.lower())
        self._call_parameters = set(inspect.signature(backbone.call).parameters)

    def prepare_inputs(self, batch: TimeSeriesBatch):
        adapt_batch = getattr(self.backbone, "adapt_batch", None)
        if adapt_batch is not None:
            return adapt_batch(batch)

        if batch.structure is not None:
            raise ValueError(f"{self.model_type} has no spatial batch adapter for {batch.arrangement.value} inputs")

        if batch.past_time_features is None and batch.future_time_features is None:
            return batch.past_values

        past_features = batch.past_time_features
        if past_features is None:
            past_features = tf.zeros([batch.batch_size, batch.context_length, 0], dtype=batch.past_values.dtype)
        values = {"x": batch.past_values, "encoder_feature": past_features}
        if batch.future_time_features is not None:
            values["decoder_feature"] = batch.future_time_features
        return values

    def _invoke(self, inputs, training=None, output_hidden_states=False):
        kwargs: Dict[str, Any] = {}
        if "training" in self._call_parameters:
            kwargs["training"] = training
        if "output_hidden_states" in self._call_parameters:
            kwargs["output_hidden_states"] = output_hidden_states
        if "return_dict" in self._call_parameters:
            kwargs["return_dict"] = True
        return self.backbone(inputs, **kwargs)

    @staticmethod
    def _tensor_from(raw, *keys):
        if isinstance(raw, dict):
            for key in keys:
                value = raw.get(key)
                if value is not None:
                    return value
            return None
        return raw

    def forward(self, batch: TimeSeriesBatch, training=None, require: OutputPort = None) -> BackboneOutput:
        inputs = self.prepare_inputs(batch)
        if require in (OutputPort.SEQUENCE, OutputPort.TEMPORAL_SEQUENCE, OutputPort.POOLED):
            if not self.capabilities.has_port(require) and not (
                require == OutputPort.POOLED and self.capabilities.has_port(OutputPort.SEQUENCE)
            ):
                raise ValueError("%s does not expose the %s output port" % (self.model_type, require.value))
            raw = self._invoke(inputs, training=training, output_hidden_states=True)
            sequence = self._tensor_from(raw, "sequence_output", "last_hidden_state", "hidden_states")
            if isinstance(sequence, (list, tuple)):
                sequence = sequence[-1]
            if sequence is None:
                raise ValueError("%s declared a sequence output but returned none" % self.model_type)
            pooled = tf.reduce_mean(sequence, axis=1) if sequence.shape.rank == 3 else sequence
            return BackboneOutput(sequence_output=sequence, pooled_output=pooled)

        raw = self._invoke(inputs, training=training)
        if isinstance(raw, dict) and "loc" in raw:
            return BackboneOutput(
                native_forecast=raw["loc"],
                distribution_params={key: value for key, value in raw.items() if tf.is_tensor(value)},
            )
        prediction = self._tensor_from(raw, "predictions", "prediction", "native_forecast", "output")
        if prediction is None:
            raise ValueError("%s returned no native forecast" % self.model_type)
        return BackboneOutput(native_forecast=prediction)
