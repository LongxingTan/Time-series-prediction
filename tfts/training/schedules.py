"""Epoch schedules for training feedback; independent of decoder execution."""

import tensorflow as tf

__all__ = ["TeacherForcingSchedule", "teacher_forcing_decay", "annealed_noise_std"]


class TeacherForcingSchedule(tf.keras.callbacks.Callback):
    """Update a task model's teacher probability outside its forward pass."""

    def __init__(self, initial, final, decay_steps):
        super().__init__()
        self.initial = float(initial)
        self.final = float(final)
        self.decay_steps = max(1, int(decay_steps))

    def on_train_batch_begin(self, batch, logs=None):
        step = tf.cast(self.model.optimizer.iterations, tf.float32)
        fraction = tf.minimum(step / self.decay_steps, 1.0)
        value = self.initial + fraction * (self.final - self.initial)
        self.model.teacher_probability.assign(value)


def _linear_schedule(epoch, warmup_epochs, total_epochs, start, end):
    epoch = max(1, int(epoch))
    fraction = min(1.0, max(0.0, (epoch - int(warmup_epochs)) / max(1, int(total_epochs) - int(warmup_epochs))))
    return float(start) + fraction * (float(end) - float(start))


def teacher_forcing_decay(
    epoch: int,
    warmup_epochs: int = 3,
    total_epochs: int = 40,
    end_teacher: float = 0.2,
) -> float:
    """Anneal the teacher-forcing probability from 1.0 down to ``end_teacher``.

    Epochs 1..``warmup_epochs`` stay fully teacher-forced (1.0); after warmup the
    probability of feeding the true target decays linearly to ``end_teacher`` by the
    final epoch, so the model increasingly relies on its own sampled feed-back —
    mirroring the inference-time distribution. 1.0 = full teacher forcing, 0.0 = fully
    autoregressive (own samples only).
    """
    return _linear_schedule(epoch, warmup_epochs, total_epochs, 1.0, end_teacher)


def annealed_noise_std(
    epoch: int,
    warmup_epochs: int = 3,
    total_epochs: int = 40,
    start: float = 0.05,
    end: float = 0.5,
) -> float:
    """Linear anneal of ``noise_std`` from ``start`` (during warmup) to ``end``.

    Epochs 1..``warmup_epochs`` stay at ``start`` (the model first learns the clean
    one-step conditional); afterwards the injected noise grows linearly so that the
    final epoch noise is ``end``. This matches the schedule that produced the best
    DeepAR generative result in this repo.
    """
    return _linear_schedule(epoch, warmup_epochs, total_epochs, start, end)
