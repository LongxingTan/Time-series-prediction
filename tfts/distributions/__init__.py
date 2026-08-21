"""Probabilistic output distributions for tfts generative models.

The distribution head is the second stage of the 5-stage pipeline:

    model architecture -> predictive distribution -> autoregressive
    generator -> sample aggregation -> inverse transformation

Each ``DistributionOutput`` owns the parameter layers (built once in ``__init__``
so the containing model can be serialized and weight-shared with the generator),
and provides:

  - ``parameters(hidden_states)``    project to distribution parameters dict
  - ``mean(parameters)``             deterministic point (for greedy decoding)
  - ``sample(parameters)``           one draw (for ancestral decoding)
  - ``loss(y_true, parameters)``     negative log-likelihood (training objective)
"""

from .base import DistributionOutput
from .normal import NormalOutput
from .outputs import DistributionOutputs

__all__ = ["DistributionOutput", "NormalOutput", "DistributionOutputs"]
