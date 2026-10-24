"""Synthetic long-format data and independent sine forecast fixtures."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ar(
    n_series: int = 10,
    timesteps: int = 400,
    seasonality: float = 3.0,
    trend: float = 3.0,
    noise: float = 0.1,
    level: float = 1.0,
    exp: bool = False,
    seed: Optional[int] = 213,
    add_covariates: bool = False,
    return_components: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    if n_series <= 0 or timesteps <= 0:
        raise ValueError("n_series and timesteps must be positive integers")

    if noise < 0:
        raise ValueError("noise parameter must be non-negative")

    rng = np.random.default_rng(seed)

    # Sample parameters for each series
    linear_trends = rng.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = rng.normal(size=n_series)[:, None] / timesteps**2
    seasonalities = rng.normal(size=n_series)[:, None]
    levels = level * rng.normal(size=n_series)[:, None]

    # Generate time index
    x = np.arange(timesteps)[None, :]

    # Calculate trend component (linear + quadratic)
    trend_component = (x * linear_trends + x**2 * quadratic_trends) * trend

    # Calculate seasonal component
    seasonal_component = seasonalities * np.sin(2 * np.pi * seasonality * x / timesteps)

    # Combine components
    series = trend_component + seasonal_component

    # Apply level scaling
    series = levels + series

    # Add noise
    series = series * (1 + noise * rng.normal(size=series.shape))

    # Apply exponential transform if requested
    if exp:
        series = np.exp(series)

    # Create DataFrame
    data = (
        pd.DataFrame(series)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "series", "level_1": "time_idx", 0: "value"})
    )

    # Add covariates if requested
    if add_covariates:
        # Add day of week (assuming each timestep is a day)
        data["day_of_week"] = data["time_idx"] % 7

        # Add month of year (assuming each timestep is a day)
        data["month"] = data["time_idx"] % 365 // 30 + 1

        # Add a categorical variable
        data["category"] = np.random.choice(["A", "B", "C"], size=len(data))

        # Add a binary variable that changes with time
        data["special_event"] = (np.sin(2 * np.pi * data["time_idx"] / 20) > 0.8).astype(int)

    # Prepare components dictionary if return_components is True
    components = {
        "linear_trends": linear_trends,
        "quadratic_trends": quadratic_trends,
        "seasonalities": seasonalities,
        "levels": levels,
        "series": series,
    }

    if return_components:
        return data, components
    else:
        return data


def load_sine(n_series=100, timesteps=32, seed=None):
    """Return unscaled sine histories as (series, time_idx, value) rows."""
    if n_series < 1 or timesteps < 1:
        raise ValueError("n_series and timesteps must be positive")
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=n_series)
    values = np.sin(np.linspace(0, 3 * np.pi, timesteps)[None, :] + phase[:, None])
    return pd.DataFrame(
        {
            "series": np.repeat(np.arange(n_series), timesteps),
            "time_idx": np.tile(np.arange(timesteps), n_series),
            "value": values.ravel(),
        }
    )
