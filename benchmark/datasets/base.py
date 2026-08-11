"""Base dataset utilities for the benchmark system."""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def split_train_valid(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split arrays into train/validation.

    Args:
        x: Input array.
        y: Target array.
        test_size: Fraction of data to use for validation.

    Returns:
        (x_train, y_train), (x_valid, y_valid).
    """
    n = len(x)
    split_idx = int(n * (1 - test_size))
    return (x[:split_idx], y[:split_idx]), (x[split_idx:], y[split_idx:])
