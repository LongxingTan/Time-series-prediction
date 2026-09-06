"""Structured outputs for forecast generation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Union

import numpy as np
import tensorflow as tf


@dataclass
class GenerationOutput:
    """Result of a ``generate(...)`` call.

    Shapes (assuming a scalar target, ``target_dim == 1``):

    - ``predictions``: ``[batch, horizon, target_dim]``
    - ``samples``:     ``[batch, num_samples, horizon, target_dim]`` (when requested)
    - ``loc``/``scale``: ``[batch, num_samples, horizon, target_dim]``

    All tensors are in normalized space; call ``preprocessor.inverse_transform`` on
    ``predictions`` to recover original units.
    """

    predictions: Union[tf.Tensor, np.ndarray]
    samples: Optional[Union[tf.Tensor, np.ndarray]] = None
    loc: Optional[Union[tf.Tensor, np.ndarray]] = None
    scale: Optional[Union[tf.Tensor, np.ndarray]] = None
    distribution_params: Optional[Mapping[str, tf.Tensor]] = None
    quantile_values: Optional[Union[tf.Tensor, np.ndarray]] = None
    values_processed: bool = False

    def numpy(self) -> "ForecastGenerationOutput":
        """Return a copy with all tensors converted via ``.numpy()`` (eager only)."""

        def _to(t: Optional[Union[tf.Tensor, np.ndarray]]):
            if t is None:
                return None
            return t.numpy() if isinstance(t, tf.Tensor) else t

        assert self.predictions is not None
        return type(self)(
            predictions=_to(self.predictions),
            samples=_to(self.samples),
            loc=_to(self.loc),
            scale=_to(self.scale),
            distribution_params=(
                None
                if self.distribution_params is None
                else {name: _to(value) for name, value in self.distribution_params.items()}
            ),
            quantile_values=_to(self.quantile_values),
            values_processed=self.values_processed,
        )


@dataclass
class ForecastGenerationOutput(GenerationOutput):
    """Forecast-specific specialization of the generic generation result."""
