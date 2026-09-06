"""Strategy-owned construction of one model step and its next input."""

from dataclasses import dataclass
from typing import Callable


def identity_input(current, value, *, offset):
    return value


@dataclass
class StepDecoder:
    step: Callable
    next_input: Callable = identity_input
