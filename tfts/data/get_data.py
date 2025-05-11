"""Generate the example data script
- https://github.com/keras-team/keras/blob/v3.3.3/keras/src/utils/file_utils.py#L130-L327
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from tfts.constants import TFTS_ASSETS_CACHE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AIR_PASSENGER_URL = (
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)


def get_data(
    name: str = "sine", train_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.1, **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    assert (test_size >= 0) & (test_size <= 1), "test_size is the ratio of test dataset"
    if name == "sine":
        return get_sine(train_length, predict_sequence_length, test_size=test_size)

    elif name == "airpassengers":
        return get_air_passengers(train_length, predict_sequence_length, test_size=test_size)

    elif name == "ar":
        return get_ar_data(**kwargs)
    else:
        raise ValueError(f"unsupported data of {name} yet, try 'sine', 'airpassengers'")


def get_sine(
    train_sequence_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.2, n_examples: int = 100
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate synthetic sine wave data.

    Parameters:
    train_sequence_length (int): Length of the training sequence.
    predict_sequence_length (int): Length of the prediction sequence.
    test_size (float): Fraction of the data to use for validation.
    n_examples (int): Number of examples to generate.

    Returns:
    (tuple): Two tuples of numpy arrays containing training and validation data.
    """
    x: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for _ in range(n_examples):
        rand = random.random() * 2 * np.pi
        sig1 = np.sin(np.linspace(rand, 3.0 * np.pi + rand, train_sequence_length + predict_sequence_length))
        sig2 = np.cos(np.linspace(rand, 3.0 * np.pi + rand, train_sequence_length + predict_sequence_length))

        x1 = sig1[:train_sequence_length]
        y1 = sig1[train_sequence_length:]
        x2 = sig2[:train_sequence_length]
        y2 = sig2[train_sequence_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])

        x.append(x_.T)
        y.append(y_.T)

    x_array = np.array(x)[:, :, 0:1]
    y_array = np.array(y)[:, :, 0:1]
    logging.info("Load sine data", x_array.shape, y_array.shape)

    if test_size > 0:
        slice = int(n_examples * (1 - test_size))
        x_train = x_array[:slice]
        y_train = y_array[:slice]
        x_valid = x_array[slice:]
        y_valid = y_array[slice:]
        return (x_train, y_train), (x_valid, y_valid)
    return x_array, y_array


def get_air_passengers(train_sequence_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.2):
    """
    A function that loads and preprocesses the air passenger data.

    Args:
        train_sequence_length (int): The length of each input sequence.
        predict_sequence_length (int): The length of each output sequence.
        test_size (float): The fraction of the data to use for validation.

    Returns:
        Tuple of training and validation data, each containing inputs and outputs.

    """
    df = pd.read_csv(AIR_PASSENGER_URL, parse_dates=None, date_parser=None, nrows=144)
    v = df.iloc[:, 1:2].values
    v = (v - np.max(v)) / (np.max(v) - np.min(v))  # MinMaxScaler

    x: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for seq in range(1, train_sequence_length + 1):
        x_roll = np.roll(v, seq, axis=0)
        x.append(x_roll)
    x_array = np.stack(x, axis=1)
    x_array = x_array[train_sequence_length:-predict_sequence_length, ::-1, :]

    for seq in range(predict_sequence_length):
        y_roll = np.roll(v, -seq)
        y.append(y_roll)
    y_array = np.stack(y, axis=1)
    y_array = y_array[train_sequence_length:-predict_sequence_length]
    logging.info("Load air passenger data", x_array.shape, y_array.shape)

    if test_size > 0:
        slice = int(len(x_array) * (1 - test_size))
        x_train = x_array[:slice]
        y_train = y_array[:slice]
        x_valid = x_array[slice:]
        y_valid = y_array[slice:]
        return (x_train, y_train), (x_valid, y_valid)
    return x_array, y_array


def get_stock_data(ticker: str = "NVDA", start_date="2023-09-01", end_date="2024-03-15") -> pd.DataFrame:
    """
    Retrieve historical stock data for a given ticker symbol.
    """
    # Download data
    import yfinance as yf

    try:
        logger.info(f"Retrieving data for {ticker} from {start_date} to {end_date}")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}")
            raise ValueError(f"No data available for ticker: {ticker}")

        logger.info(f"Successfully retrieved {len(data)} records for {ticker}")
        return data

    except Exception as e:
        logger.exception("Stock data retrieval failed.")
        raise RuntimeError(f"Failed to fetch stock data: {e}")


def get_ar_data(
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

    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Sample parameters for each series
    linear_trends = np.random.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = np.random.normal(size=n_series)[:, None] / timesteps**2
    seasonalities = np.random.normal(size=n_series)[:, None]
    levels = level * np.random.normal(size=n_series)[:, None]

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
    series = series * (1 + noise * np.random.normal(size=series.shape))

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
