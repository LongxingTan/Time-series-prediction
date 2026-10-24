from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import tensorflow as tf

from tfts.models.base import BackboneOutput, ModelOutput


@dataclass(frozen=True)
class ForecastOutput(ModelOutput):
    predictions: Optional[tf.Tensor] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[tf.Tensor] = None
    quantiles: Optional[Tuple[float, ...]] = None
    samples: Optional[tf.Tensor] = None
    backbone_output: Optional[BackboneOutput] = None

    def prediction(self):
        """Return the task's primary prediction tensor."""
        return self.predictions
