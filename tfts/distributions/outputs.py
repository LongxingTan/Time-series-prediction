"""Structured container for distribution parameters projected by a ``DistributionOutput``."""

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf


@dataclass
class DistributionOutputs:
    """Typed view over a dict of distribution parameter tensors.

    Shapes follow ``(batch, seq, target_dim)``. The ``as_dict()`` view is what the
    model returns from its forward ``__call__`` so the existing ``{"loc","scale"}``
    output contract is preserved unchanged.
    """

    loc: tf.Tensor
    scale: tf.Tensor

    def as_dict(self) -> Dict[str, tf.Tensor]:
        return {"loc": self.loc, "scale": self.scale}

    @classmethod
    def from_dict(cls, params: Dict[str, tf.Tensor]) -> "DistributionOutputs":
        return cls(loc=params["loc"], scale=params["scale"])
