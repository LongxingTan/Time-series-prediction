"""Time related feature: date, holiday, event

This module provides functionality to generate various datetime-based features from time series data.
It supports basic datetime features (hour, day, month, etc.), cyclical features, and holiday features.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .registry import FeatureRegistry, registry

logger = logging.getLogger(__name__)


def _get_cyclical_features(dt: pd.Series, period: int, max_val: int, feature_name: str) -> pd.DataFrame:
    """Convert a datetime feature into cyclical features using sine and cosine.

    Args:
        dt: Series of datetime values
        period: The period of the cycle (e.g., 24 for hours, 7 for days of week)
        max_val: The maximum value in the cycle (e.g., 23 for hours, 6 for days of week)
        feature_name: Name of the feature (e.g., 'month', 'dayofweek', 'hour')

    Returns:
        DataFrame with sine and cosine features
    """
    sin_feature = np.sin(2 * np.pi * dt / period)
    cos_feature = np.cos(2 * np.pi * dt / period)
    return pd.DataFrame({f"{feature_name}_sin": sin_feature, f"{feature_name}_cos": cos_feature})


@registry
def add_datetime_feature(
    data: pd.DataFrame,
    time_col: str,
    features: Optional[List[str]] = None,
    holiday_calendar: Optional[USFederalHolidayCalendar] = None,
    custom_holidays: Optional[List[pd.Timestamp]] = None,
    business_days: bool = False,
) -> pd.DataFrame:
    """Add datetime features to the dataframe.

    This function adds various datetime-based features to the input dataframe.
    It supports basic datetime features, cyclical features, and holiday features.

    Args:
        data: Input DataFrame containing the time column
        time_col: Name of the column containing datetime values
        features: List of features to generate. If None, generates all features.
            Supported features:
            - 'year': Year of the date
            - 'quarter': Quarter of the year (1-4)
            - 'month': Month of the year (1-12)
            - 'week': Week of the year (1-52)
            - 'day': Day of the month (1-31)
            - 'dayofweek': Day of the week (0-6, Monday=0)
            - 'dayofyear': Day of the year (1-365)
            - 'hour': Hour of the day (0-23)
            - 'minute': Minute of the hour (0-59)
            - 'second': Second of the minute (0-59)
            - 'is_month_start': Whether the date is the first day of the month
            - 'is_month_end': Whether the date is the last day of the month
            - 'is_quarter_start': Whether the date is the first day of the quarter
            - 'is_quarter_end': Whether the date is the last day of the quarter
            - 'is_year_start': Whether the date is the first day of the year
            - 'is_year_end': Whether the date is the last day of the year
            - 'is_weekend': Whether the date is a weekend
            - 'is_holiday': Whether the date is a holiday
            - 'is_business_day': Whether the date is a business day
            - 'month_sin', 'month_cos': Cyclical features for month
            - 'dayofweek_sin', 'dayofweek_cos': Cyclical features for day of week
            - 'hour_sin', 'hour_cos': Cyclical features for hour
        holiday_calendar: Calendar to use for holiday detection. If None, uses USFederalHolidayCalendar
        custom_holidays: List of additional holidays to consider
        business_days: Whether to add business day features

    Returns:
        DataFrame with added datetime features

    Raises:
        ValueError: If time_col is not in the dataframe
        TypeError: If time_col is not datetime type
    """
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not in dataframe")

    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
        try:
            data[time_col] = pd.to_datetime(data[time_col])
        except Exception as e:
            raise TypeError(f"Could not convert {time_col} to datetime: {str(e)}")

    # Define valid features
    valid_features = {
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "dayofyear",
        "hour",
        "minute",
        "second",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_weekend",
        "is_holiday",
        "is_business_day",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "hour_sin",
        "hour_cos",
    }

    if features is None:
        features = list(valid_features)
    else:
        invalid_features = [f for f in features if f not in valid_features]
        if invalid_features:
            raise KeyError(f"Invalid feature names: {invalid_features}. Valid features are: {sorted(valid_features)}")

    dt = data[time_col]
    new_features = pd.DataFrame(index=data.index)

    # Basic datetime features
    if "year" in features:
        new_features["year"] = dt.dt.year
    if "quarter" in features:
        new_features["quarter"] = dt.dt.quarter
    if "month" in features:
        new_features["month"] = dt.dt.month
    if "week" in features:
        new_features["week"] = dt.dt.isocalendar().week
    if "day" in features:
        new_features["day"] = dt.dt.day
    if "dayofweek" in features:
        new_features["dayofweek"] = dt.dt.dayofweek
    if "dayofyear" in features:
        new_features["dayofyear"] = dt.dt.dayofyear
    if "hour" in features:
        new_features["hour"] = dt.dt.hour
    if "minute" in features:
        new_features["minute"] = dt.dt.minute
    if "second" in features:
        new_features["second"] = dt.dt.second

    # Boolean features
    if "is_month_start" in features:
        new_features["is_month_start"] = dt.dt.is_month_start.astype(int)
    if "is_month_end" in features:
        new_features["is_month_end"] = dt.dt.is_month_end.astype(int)
    if "is_quarter_start" in features:
        new_features["is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    if "is_quarter_end" in features:
        new_features["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    if "is_year_start" in features:
        new_features["is_year_start"] = dt.dt.is_year_start.astype(int)
    if "is_year_end" in features:
        new_features["is_year_end"] = dt.dt.is_year_end.astype(int)
    if "is_weekend" in features:
        new_features["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    # Holiday features
    if "is_holiday" in features or "is_business_day" in features:
        if holiday_calendar is None:
            holiday_calendar = USFederalHolidayCalendar()

        holidays = holiday_calendar.holidays(start=dt.min(), end=dt.max())
        if custom_holidays:
            holidays = holidays.union(pd.DatetimeIndex(custom_holidays))

        if "is_holiday" in features:
            new_features["is_holiday"] = dt.isin(holidays).astype(int)

        if "is_business_day" in features or business_days:
            business_day = CustomBusinessDay(calendar=holiday_calendar)
            business_days = pd.date_range(start=dt.min(), end=dt.max(), freq=business_day)
            new_features["is_business_day"] = dt.isin(business_days).astype(int)

    # Cyclical features
    if "month_sin" in features or "month_cos" in features:
        month_cyclic = _get_cyclical_features(dt.dt.month, 12, 11, "month")
        new_features = pd.concat([new_features, month_cyclic], axis=1)

    if "dayofweek_sin" in features or "dayofweek_cos" in features:
        dow_cyclic = _get_cyclical_features(dt.dt.dayofweek, 7, 6, "dayofweek")
        new_features = pd.concat([new_features, dow_cyclic], axis=1)

    if "hour_sin" in features or "hour_cos" in features:
        hour_cyclic = _get_cyclical_features(dt.dt.hour, 24, 23, "hour")
        new_features = pd.concat([new_features, hour_cyclic], axis=1)

    # Add prefix to all new features
    new_features.columns = [f"{time_col}_{col}" for col in new_features.columns]

    # Combine with original data
    result = pd.concat([data, new_features], axis=1)

    logger.info(f"Added {len(new_features.columns)} datetime features")
    return result
