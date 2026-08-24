"""Loss functions for time series prediction."""

from .loss import MultiQuantileLoss, smape_loss

__all__ = ["MultiQuantileLoss", "smape_loss"]
