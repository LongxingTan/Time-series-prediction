"""Auto regression feature: lag and roll

This module provides functionality to generate various one-order features for time series data.
It includes lag features, rolling window features, and basic transformations.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .registry import FeatureRegistry, registry

logger = logging.getLogger(__name__)


@registry
def add_lag_feature(
    data: pd.DataFrame,
    columns: Union[str, List[str]],
    lags: Union[int, List[int]],
    time_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Add lag features to the dataframe.

    Args:
        data: Input DataFrame
        columns: Column(s) to create lag features for
        lags: Number of periods to lag. Can be a single integer or a list of integers
        time_col: Column containing time information. If None, assumes data is already sorted
        group_cols: Columns to group by before creating lags
        fill_method: Method to fill missing values. Options: 'ffill', 'bfill', 'interpolate'

    Returns:
        DataFrame with added lag features

    Raises:
        ValueError: If columns or lags are invalid
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(lags, int):
        lags = [lags]

    if not all(col in data.columns for col in columns):
        raise ValueError("All columns must exist in the dataframe")

    if not all(isinstance(lag, int) and lag > 0 for lag in lags):
        raise ValueError("All lags must be positive integers")

    result = data.copy()

    # Sort by time if time_col is provided
    if time_col is not None:
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not in dataframe")
        result = result.sort_values(by=time_col)

    # Create lag features
    for col in columns:
        for lag in lags:
            if group_cols is not None:
                # Create lag features within each group
                result[f"{col}_lag_{lag}"] = result.groupby(group_cols)[col].shift(lag)
            else:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)

    # Fill missing values
    if fill_method is not None:
        if fill_method == "ffill":
            result = result.ffill()
        elif fill_method == "bfill":
            result = result.bfill()
        elif fill_method == "interpolate":
            result = result.interpolate()
        else:
            raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'interpolate'")

    logger.info(f"Added {len(columns) * len(lags)} lag features")
    return result


@registry
def add_roll_feature(
    data: pd.DataFrame,
    columns: Union[str, List[str]],
    windows: Union[int, List[int]],
    functions: Optional[Union[str, List[str]]] = None,
    time_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Add rolling window features to the dataframe.

    Args:
        data: Input DataFrame
        columns: Column(s) to create rolling features for
        windows: Window size(s) for rolling calculations
        functions: Function(s) to apply. If None, uses ['mean', 'std', 'min', 'max']
        time_col: Column containing time information. If None, assumes data is already sorted
        group_cols: Columns to group by before creating rolling features
        min_periods: Minimum number of observations in window required to have a value

    Returns:
        DataFrame with added rolling features

    Raises:
        ValueError: If columns, windows, or functions are invalid
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(windows, int):
        windows = [windows]
    if functions is None:
        functions = ["mean", "std", "min", "max"]
    if isinstance(functions, str):
        functions = [functions]

    if not all(col in data.columns for col in columns):
        raise ValueError("All columns must exist in the dataframe")

    if not all(isinstance(window, int) and window > 0 for window in windows):
        raise ValueError("All windows must be positive integers")

    valid_functions = ["mean", "std", "min", "max", "median", "skew", "kurt"]
    if not all(func in valid_functions for func in functions):
        raise ValueError(f"Functions must be one of: {valid_functions}")

    result = data.copy()

    # Sort by time if time_col is provided
    if time_col is not None:
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not in dataframe")
        result = result.sort_values(by=time_col)

    # Create rolling features
    for col in columns:
        for window in windows:
            if group_cols is not None:
                # Create rolling features within each group
                group = result.groupby(group_cols)[col]
                for func in functions:
                    if func == "mean":
                        result[f"{col}_roll_{window}_mean"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).mean()
                        )
                    elif func == "std":
                        result[f"{col}_roll_{window}_std"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).std()
                        )
                    elif func == "min":
                        result[f"{col}_roll_{window}_min"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).min()
                        )
                    elif func == "max":
                        result[f"{col}_roll_{window}_max"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).max()
                        )
                    elif func == "median":
                        result[f"{col}_roll_{window}_median"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).median()
                        )
                    elif func == "skew":
                        result[f"{col}_roll_{window}_skew"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).skew()
                        )
                    elif func == "kurt":
                        result[f"{col}_roll_{window}_kurt"] = group.transform(
                            lambda x: x.rolling(window, min_periods=min_periods).kurt()
                        )
            else:
                for func in functions:
                    if func == "mean":
                        result[f"{col}_roll_{window}_mean"] = (
                            result[col].rolling(window, min_periods=min_periods).mean()
                        )
                    elif func == "std":
                        result[f"{col}_roll_{window}_std"] = result[col].rolling(window, min_periods=min_periods).std()
                    elif func == "min":
                        result[f"{col}_roll_{window}_min"] = result[col].rolling(window, min_periods=min_periods).min()
                    elif func == "max":
                        result[f"{col}_roll_{window}_max"] = result[col].rolling(window, min_periods=min_periods).max()
                    elif func == "median":
                        result[f"{col}_roll_{window}_median"] = (
                            result[col].rolling(window, min_periods=min_periods).median()
                        )
                    elif func == "skew":
                        result[f"{col}_roll_{window}_skew"] = (
                            result[col].rolling(window, min_periods=min_periods).skew()
                        )
                    elif func == "kurt":
                        result[f"{col}_roll_{window}_kurt"] = (
                            result[col].rolling(window, min_periods=min_periods).kurt()
                        )

    logger.info(f"Added {len(columns) * len(windows) * len(functions)} rolling features")
    return result


@registry
def add_transform_feature(
    data: pd.DataFrame,
    columns: Union[str, List[str]],
    functions: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """Add transformed features to the dataframe.

    Args:
        data: Input DataFrame
        columns: Column(s) to create transformed features for
        functions: Function(s) to apply. If None, uses ['log1p', 'sqrt', 'square']

    Returns:
        DataFrame with added transformed features

    Raises:
        ValueError: If columns or functions are invalid
    """
    if isinstance(columns, str):
        columns = [columns]
    if functions is None:
        functions = ["log1p", "sqrt", "square"]
    if isinstance(functions, str):
        functions = [functions]

    if not all(col in data.columns for col in columns):
        raise ValueError("All columns must exist in the dataframe")

    valid_functions = ["log1p", "sqrt", "square", "exp", "sin", "cos", "tan"]
    if not all(func in valid_functions for func in functions):
        raise ValueError(f"Functions must be one of: {valid_functions}")

    result = data.copy()

    # Create transformed features
    for col in columns:
        for func in functions:
            if func == "log1p":
                result[f"{col}_log1p"] = np.log1p(result[col])
            elif func == "sqrt":
                result[f"{col}_sqrt"] = np.sqrt(result[col])
            elif func == "square":
                result[f"{col}_square"] = np.square(result[col])
            elif func == "exp":
                result[f"{col}_exp"] = np.exp(result[col])
            elif func == "sin":
                result[f"{col}_sin"] = np.sin(result[col])
            elif func == "cos":
                result[f"{col}_cos"] = np.cos(result[col])
            elif func == "tan":
                result[f"{col}_tan"] = np.tan(result[col])

    logger.info(f"Added {len(columns) * len(functions)} transformed features")
    return result


@registry
def add_moving_average_feature(
    data: pd.DataFrame,
    columns: Union[str, List[str]],
    windows: Union[int, List[int]],
    time_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Add moving average features to the dataframe.

    This is a specialized version of rolling features that focuses on different types
    of moving averages.

    Args:
        data: Input DataFrame
        columns: Column(s) to create moving average features for
        windows: Window size(s) for moving average calculations
        time_col: Column containing time information. If None, assumes data is already sorted
        group_cols: Columns to group by before creating moving averages
        min_periods: Minimum number of observations in window required to have a value

    Returns:
        DataFrame with added moving average features

    Raises:
        ValueError: If columns or windows are invalid
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(windows, int):
        windows = [windows]

    if not all(col in data.columns for col in columns):
        raise ValueError("All columns must exist in the dataframe")

    if not all(isinstance(window, int) and window > 0 for window in windows):
        raise ValueError("All windows must be positive integers")

    result = data.copy()

    if time_col is not None:
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not in dataframe")
        result = result.sort_values(by=time_col)

    # Create moving average features
    for col in columns:
        for window in windows:
            if group_cols is not None:
                # Create moving averages within each group
                group = result.groupby(group_cols)[col]
                # Simple Moving Average (SMA)
                result[f"{col}_sma_{window}"] = group.transform(
                    lambda x: x.rolling(window, min_periods=min_periods).mean()
                )
                # Exponential Moving Average (EMA)
                result[f"{col}_ema_{window}"] = group.transform(
                    lambda x: x.ewm(span=window, min_periods=min_periods).mean()
                )
                # Weighted Moving Average (WMA)
                weights = np.arange(1, window + 1)
                result[f"{col}_wma_{window}"] = group.transform(
                    lambda x: x.rolling(window, min_periods=min_periods).apply(
                        lambda y: np.sum(weights[: len(y)] * y) / weights[: len(y)].sum(), raw=True
                    )
                )
                # Hull Moving Average (HMA)
                wma1 = group.transform(
                    lambda x: x.rolling(window // 2, min_periods=min_periods).apply(
                        lambda y: np.sum(weights[: len(y)] * y) / weights[: len(y)].sum(), raw=True
                    )
                )
                wma2 = group.transform(
                    lambda x: x.rolling(window, min_periods=min_periods).apply(
                        lambda y: np.sum(weights[: len(y)] * y) / weights[: len(y)].sum(), raw=True
                    )
                )
                hull = 2 * wma1 - wma2
                result[f"{col}_hma_{window}"] = hull.rolling(int(np.sqrt(window)), min_periods=min_periods).mean()
            else:
                # Simple Moving Average (SMA)
                result[f"{col}_sma_{window}"] = result[col].rolling(window, min_periods=min_periods).mean()
                # Exponential Moving Average (EMA)
                result[f"{col}_ema_{window}"] = result[col].ewm(span=window, min_periods=min_periods).mean()
                # Weighted Moving Average (WMA)
                weights = np.arange(1, window + 1)
                result[f"{col}_wma_{window}"] = (
                    result[col]
                    .rolling(window, min_periods=min_periods)
                    .apply(lambda x: np.sum(weights[: len(x)] * x) / weights[: len(x)].sum(), raw=True)
                )
                # Hull Moving Average (HMA)
                wma1 = (
                    result[col]
                    .rolling(window // 2, min_periods=min_periods)
                    .apply(lambda x: np.sum(weights[: len(x)] * x) / weights[: len(x)].sum(), raw=True)
                )
                wma2 = (
                    result[col]
                    .rolling(window, min_periods=min_periods)
                    .apply(lambda x: np.sum(weights[: len(x)] * x) / weights[: len(x)].sum(), raw=True)
                )
                hull = 2 * wma1 - wma2
                result[f"{col}_hma_{window}"] = hull.rolling(int(np.sqrt(window)), min_periods=min_periods).mean()

    logger.info(f"Added {len(columns) * len(windows) * 4} moving average features")
    return result
