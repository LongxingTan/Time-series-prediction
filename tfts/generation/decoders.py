"""The single execution contract for direct, recursive, and native decoding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol

import tensorflow as tf

from tfts.registry import register_decoder

from .chunk import Chunk, State


@dataclass(frozen=True)
class _DecodeSession:
    context: Any
    state: Any
    previous: tf.Tensor


@dataclass(frozen=True)
class _StepOutput:
    prediction: tf.Tensor
    state: Any = None
    distribution: Optional[Any] = None
    parameters: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None


def _empty_prefix(prediction):
    return prediction[:, :0, ...]


class Decoder(Protocol):
    output_chunk_length: int

    def start(self, batch, *, horizon, training=False) -> State: ...  # noqa: E704

    def step(self, state, *, offset, training=False) -> Chunk: ...  # noqa: E704

    def feed(self, state, feedback, *, offset) -> State: ...  # noqa: E704


@register_decoder("direct")
class DirectDecoder:
    def __init__(self, model, horizon=None):
        self.model = model
        self.output_chunk_length = int(model.output_chunk_length)

    def start(self, batch, *, horizon, training=False):
        clean = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        output = self.model.forward(clean, training=training)
        return State({"batch": clean, "output": output})

    def step(self, state, *, offset, training=False):
        if tf.get_static_value(offset) not in (None, 0):
            raise RuntimeError("DirectDecoder emits exactly one chunk")
        batch, output = state.value["batch"], state.value["output"]
        return Chunk(
            offset=tf.cast(offset, tf.int32),
            past=batch.past_values,
            generated=_empty_prefix(output.predictions),
            prediction=output.predictions,
            parameters=output.distribution_params,
            quantiles=output.quantile_values,
        )

    def feed(self, state, feedback, *, offset):
        raise NotImplementedError("DirectDecoder never feeds a second chunk")


@register_decoder("recursive")
class RecursiveDecoder:
    output_chunk_length = 1

    def __init__(self, model, horizon=None):
        self.model = model

    def start(self, batch, *, horizon, training=False):
        clean = replace(batch, future_values=None, future_observed_mask=None, labels=None)
        return State({"batch": clean})

    def step(self, state, *, offset, training=False):
        batch = state.value["batch"]
        output = self.model.forward(batch, training=training)
        prediction = output.predictions[:, :1, ...]
        parameters = None
        if output.distribution_params is not None:
            parameters = tf.nest.map_structure(lambda value: value[:, :1, ...], output.distribution_params)
        quantiles = None if output.quantile_values is None else output.quantile_values[:, :1, ...]
        return Chunk(
            offset=tf.cast(offset, tf.int32),
            past=batch.past_values,
            generated=_empty_prefix(prediction),
            prediction=prediction,
            parameters=parameters,
            quantiles=quantiles,
        )

    def feed(self, state, feedback, *, offset):
        return State({"batch": state.value["batch"].advance(feedback, offset=offset)})


@register_decoder("native")
class NativeDecoder:
    output_chunk_length = 1

    def __init__(self, model, horizon=None):
        if getattr(model, "head", None) is not None:
            raise ValueError("NativeDecoder cannot bypass a learned task head")
        self.model = model
        self.backbone = model.backbone

    def start(self, batch, *, horizon, training=False):
        model_batch, restore = self.model.prepare_backbone_batch(batch)
        session = self.backbone.initialize_decode(model_batch, horizon=horizon, training=training)
        return State(
            {
                "batch": model_batch,
                "context": session.context,
                "decoder_state": session.state,
                "previous": session.previous,
                "restore": restore,
            }
        )

    def step(self, state, *, offset, training=False):
        values = state.value
        output = self.backbone.decode_step(
            values["previous"],
            values["decoder_state"],
            values["context"],
            offset=offset,
            training=training,
        )
        values["next_decoder_state"] = output.state
        prediction = values["restore"](output.prediction)
        parameters = output.parameters
        if parameters is not None:
            parameters = values["restore"](parameters)
        return Chunk(
            offset=tf.cast(offset, tf.int32),
            past=values["batch"].past_values,
            generated=_empty_prefix(prediction),
            prediction=prediction,
            parameters=parameters,
            quantiles=output.quantile_values,
        )

    def feed(self, state, feedback, *, offset):
        values = state.value
        values["previous"] = self.backbone.next_input(feedback, values["context"], offset=offset)
        values["decoder_state"] = values.pop("next_decoder_state")
        return state


def decoder_for(model, mode="auto", *, horizon):
    output_chunk_length = int(model.output_chunk_length)
    native = all(hasattr(model.backbone, name) for name in ("initialize_decode", "decode_step", "next_input"))
    learned_head = getattr(model, "head", None) is not None

    if mode == "auto":
        if horizon <= output_chunk_length:
            mode = "direct"
        elif native and not learned_head:
            mode = "native"
        else:
            mode = "recursive"
    if mode == "native" and learned_head:
        raise ValueError("a task model with a learned head cannot use NativeDecoder")
    if mode == "native" and not native:
        raise ValueError("this backbone does not implement native decoding")
    if mode == "direct" and horizon > output_chunk_length:
        raise ValueError("direct mode requires horizon <= output_chunk_length")
    if mode not in {"direct", "recursive", "native"}:
        raise ValueError(f"Unknown generation mode {mode!r}")
    decoder_type = {"direct": DirectDecoder, "recursive": RecursiveDecoder, "native": NativeDecoder}[mode]
    return decoder_type(model=model, horizon=horizon)


__all__ = [
    "Decoder",
    "DirectDecoder",
    "NativeDecoder",
    "RecursiveDecoder",
    "decoder_for",
]
