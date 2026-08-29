"""Adapters between the canonical batch and architecture implementations."""

from __future__ import annotations

import inspect
from typing import Any, Dict

import tensorflow as tf

from tfts.contracts import BackboneOutput, OutputPort, TimeSeriesBatch


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
            raise ValueError(f"{self.model_type} has no spatial batch adapter for {batch.layout.value} inputs")

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
