"""
`ForecastPFN: Synthetically-Trained Zero-Shot Forecasting
<https://arxiv.org/abs/2311.01933>`_

TODO: Implement ForecastPFN model. This is currently a stub.
"""

import logging
from typing import Optional

from .base import BaseModel, CommonConfig

logger = logging.getLogger(__name__)


class PFNConfig(CommonConfig):
    model_type: str = "pfn"

    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)
        logger.warning("PFNConfig is a stub — the PFN model is not yet implemented.")


class PFN(BaseModel):
    """ForecastPFN model — Not yet implemented."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[PFNConfig] = None):
        super().__init__()
        raise NotImplementedError(
            "PFN (ForecastPFN) is not implemented yet. "
            "See https://github.com/LongxingTan/Time-series-prediction/issues for progress."
        )
