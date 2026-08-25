"""Distribution-agnostic iterative forecast driver."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import tensorflow as tf

from .processors import ForecastProcessorList, resolve_forecast_processors
from .strategy import FeedbackPolicy, SamplingStrategy, StepOutput, ValueFeedback


@dataclass
class RolloutOutput:
    values: tf.Tensor
    steps: List[StepOutput]
    state: Any


class GenerationEngine:
    """Run a one-step model using independently replaceable sampling and feedback."""

    def __init__(self, strategy: SamplingStrategy, feedback: Optional[FeedbackPolicy] = None, processors=None):
        self.strategy = strategy
        self.feedback = feedback or ValueFeedback()
        self.processors: ForecastProcessorList = resolve_forecast_processors(processors)

    def run(
        self,
        step_fn: Callable[[Any, Any, int], StepOutput],
        initial_input,
        initial_state,
        horizon: int,
        *,
        teacher=None,
        context=None,
        seed_for_step: Optional[Callable[[int], Any]] = None,
    ) -> RolloutOutput:
        current, model_state = initial_input, initial_state
        strategy_state = self.strategy.initialize(context=context)
        values, steps = [], []
        for step in range(int(horizon)):
            output = step_fn(current, model_state, step)
            if not isinstance(output, StepOutput):
                raise TypeError("step_fn must return StepOutput.")
            model_state = output.state
            seed = seed_for_step(step) if seed_for_step is not None else None
            selected = self.strategy.sample(output, step=step, seed=seed, teacher=teacher, state=strategy_state)
            strategy_state = selected.state
            history = tf.concat(values, axis=1) if values else None
            selected = self.processors(
                history,
                selected,
                step=step,
                step_output=output,
                context=context,
            )
            values.append(selected.value)
            steps.append(output)
            if step + 1 < horizon:
                current = self.feedback.update(current, selected, step=step, context=context)
        if not values:
            raise ValueError("horizon must be >= 1.")
        return RolloutOutput(tf.concat(values, axis=1), steps, model_state)
