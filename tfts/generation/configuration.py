"""Configuration for forecast generation."""

from dataclasses import dataclass, fields
from typing import Any, List, Optional, Union


@dataclass
class ForecastGenerationConfig:
    """Controls how an autoregressive model produces a forecast.

    This is an *optional* override of the default single-pass ``__call__`` behaviour:
    generative (sampled) forecasting is only used when the user explicitly calls
    ``model.generate(...)`` with (or without) a config.
    """

    horizon: Optional[int] = None
    """Number of generated steps. Defaults to the model prediction length."""

    mode: str = "ancestral"
    """Decoding strategy. One of ``ancestral``, ``greedy``, ``teacher_forced``, ``sample``.

    - ``ancestral``: draw from the predicted distribution and feed the sample back.
    - ``greedy``: feed ``loc`` back at every step (deterministic).
    - ``teacher_forced``: feed the supplied lagged targets back (diagnostic).
    - ``sample``: like ``ancestral`` but retains every trajectory.
    """

    num_samples: int = 100
    """Number of sampled trajectories (ancestral/sample modes). Ignored for greedy."""

    aggregation: Optional[str] = "mean"
    """How to reduce the sampled trajectories to a point forecast. ``mean``, ``median``,
    ``none`` (use the first trajectory)."""

    return_samples: bool = False
    """Whether to also return all individual trajectories in ``output.samples``."""

    seed: Optional[int] = None
    """Optional RNG seed; same seed + same inputs reproduces identical trajectories."""

    batch_size: Optional[int] = None
    """Optional chunk size over the batch for memory; seeded results are numerically
    equivalent to unchunked generation."""

    quantiles: Optional[List[float]] = None
    """Optional quantile aggregation targets (future), e.g. ``[0.1, 0.5, 0.9]``."""

    strategy: Optional[Any] = None
    """Optional SamplingStrategy instance or callable. Overrides ``mode``."""

    feedback: Optional[Any] = None
    """Optional FeedbackPolicy instance used to construct the next decoder input."""

    processors: Optional[Any] = None
    """Optional forecast processor or iterable applied to every generated value."""

    @classmethod
    def from_args(
        cls, generation_config: Optional[Union["ForecastGenerationConfig", dict]]
    ) -> "ForecastGenerationConfig":
        """Normalize ``None``/``dict``/instance input into a config."""
        if generation_config is None:
            return cls()
        if isinstance(generation_config, ForecastGenerationConfig):
            return generation_config
        if isinstance(generation_config, dict):
            known = {f.name for f in fields(cls)}
            unknown = set(generation_config) - known
            if unknown:
                raise ValueError(f"Unknown generation config fields: {sorted(unknown)}")
            return cls(**generation_config)
        raise TypeError(
            f"generation_config must be a ForecastGenerationConfig, dict, or None, got {type(generation_config)!r}"
        )
