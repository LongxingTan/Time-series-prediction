"""Opt-in recursive generation for direct forecasting models."""

from typing import Optional

import tensorflow as tf

from .configuration import ForecastGenerationConfig
from .engine import GenerationEngine
from .outputs import ForecastGenerationOutput
from .strategy import StepOutput, TargetFeedback, resolve_feedback_policy, resolve_sampling_strategy


class RollingWindowGenerationMixin:
    """Generate recursively by repeatedly calling a direct model on a rolling window.

    Models may override :meth:`prepare_rolling_inputs` or :meth:`predict_rolling_step`.
    By default the first step of a direct multi-horizon output is used at each iteration.
    """

    def prepare_rolling_inputs(self, inputs):
        if isinstance(inputs, dict):
            for key in ("encoder_feature", "x", "inputs"):
                if inputs.get(key) is not None:
                    return tf.convert_to_tensor(inputs[key])
            raise ValueError("Could not find a rolling input tensor in the input mapping.")
        if isinstance(inputs, (list, tuple)):
            # TFTS models conventionally use the merged encoder feature as item 1.
            return tf.convert_to_tensor(inputs[1] if len(inputs) > 1 else inputs[0])
        return tf.convert_to_tensor(inputs)

    def predict_rolling_step(self, window, step: int) -> tf.Tensor:
        output = self(window)
        if isinstance(output, dict):
            output = output.get("predictions", output.get("logits", output.get("loc")))
        if output is None:
            raise ValueError("The model output has no predictions, logits, or loc value.")
        output = tf.convert_to_tensor(output)
        if output.shape.rank == 2:
            output = output[:, :, None]
        return output[:, 0:1, :]

    def generate(
        self,
        inputs,
        generation_config: Optional[object] = None,
        *,
        horizon: Optional[int] = None,
        strategy=None,
        feedback=None,
        target_indices=(0,),
        **kwargs,
    ) -> ForecastGenerationOutput:
        config = ForecastGenerationConfig.from_args(generation_config)
        if kwargs:
            config = ForecastGenerationConfig.from_args({**config.__dict__, **kwargs})
        horizon = int(horizon or getattr(config, "horizon", None) or self.predict_sequence_length)
        strategy = resolve_sampling_strategy(strategy or config.strategy or "greedy")
        if feedback is None and config.feedback is None:
            feedback = TargetFeedback(target_indices)
        else:
            feedback = resolve_feedback_policy(feedback if feedback is not None else config.feedback)
        window = self.prepare_rolling_inputs(inputs)
        if window.shape.rank != 3:
            raise ValueError("Rolling generation expects [batch, time, feature] inputs.")

        def step_fn(current, state, step):
            return StepOutput(prediction=self.predict_rolling_step(current, step), state=state)

        rollout = GenerationEngine(strategy, feedback, config.processors).run(step_fn, window, None, horizon)
        return ForecastGenerationOutput(predictions=rollout.values)
