"""
`TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
<https://arxiv.org/pdf/2405.14616>`_

TODO: Implement TimeMixer model. This is currently a stub.
"""

import logging
from typing import Optional

from .base import BaseConfig, BaseModel

logger = logging.getLogger(__name__)


class TimeMixerConfig(BaseConfig):
    model_type: str = "timemixer"

    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)
        logger.warning("TimeMixerConfig is a stub — the TimeMixer model is not yet implemented.")


class TimeMixer(BaseModel):
    """TimeMixer model — Not yet implemented."""

    def __init__(self, predict_sequence_length: int = 1, config: Optional[TimeMixerConfig] = None):
        super().__init__()
        raise NotImplementedError(
            "TimeMixer is not implemented yet. "
            "See https://github.com/LongxingTan/Time-series-prediction/issues for progress."
        )
