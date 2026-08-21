"""Training-time feedback / exposure-bias schedules.

These are scaffolding for later phases of the DeepAR refactor (noisy teacher forcing,
scheduled sampling). They deliberately live OUTSIDE the data preprocessor: the
preprocessor keeps producing clean teacher-forced lagged targets, and any corruption /
feedback selection is applied at training time by the model's custom training step.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScheduleConfig:
    """Linear (or constant) schedule for a training-time knob.

    ``value = initial_value`` before ``warmup_epochs`` is reached, then ramps linearly
    toward ``final_value``. A callback updates a ``tf.Variable`` each epoch so the value
    can change without rebuilding the graph.
    """

    kind: str = "linear"
    initial_value: float = 0.0
    final_value: float = 0.5
    warmup_epochs: int = 30

    def value_at(self, epoch: int) -> float:
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return float(self.final_value)
        frac = max(0.0, (epoch / self.warmup_epochs))
        return float(self.initial_value + frac * (self.final_value - self.initial_value))


@dataclass
class FeedbackTrainingConfig:
    """Controls how decoder lagged-target feedback is formed during training.

    Strategies:

    - ``teacher_forcing``: current default (feed true lagged targets).
    - ``noisy_teacher_forcing``: add noise (``noise_schedule``) to lagged targets.
    - ``scheduled_sampling``: probabilistically replace true lagged targets with model
      output (``sampling_schedule`` gives the replacement probability).
    - ``mixed``: scheduled sampling plus feedback noise.
    """

    strategy: str = "teacher_forcing"
    sampling_schedule: Optional[ScheduleConfig] = None
    noise_schedule: Optional[ScheduleConfig] = None
    feedback_value: str = "sample"

    def noise_std_at(self, epoch: int) -> float:
        return self.noise_schedule.value_at(epoch) if self.noise_schedule is not None else 0.0

    def sampling_prob_at(self, epoch: int) -> float:
        return self.sampling_schedule.value_at(epoch) if self.sampling_schedule is not None else 0.0
