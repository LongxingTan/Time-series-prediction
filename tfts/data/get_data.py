"""Generate the example data script
- https://github.com/keras-team/keras/blob/v3.3.3/keras/src/utils/file_utils.py#L130-L327
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, get_file

from tfts.constants import TFTS_DATASETS_CACHE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AIR_PASSENGERS_VALUES = np.array(
    [
        112,
        118,
        132,
        129,
        121,
        135,
        148,
        148,
        136,
        119,
        104,
        118,
        115,
        126,
        141,
        135,
        125,
        149,
        170,
        170,
        158,
        133,
        114,
        140,
        145,
        150,
        178,
        163,
        172,
        178,
        199,
        199,
        184,
        162,
        146,
        166,
        171,
        180,
        193,
        181,
        183,
        218,
        230,
        242,
        209,
        191,
        172,
        194,
        196,
        196,
        236,
        235,
        229,
        243,
        264,
        272,
        237,
        211,
        180,
        201,
        204,
        188,
        235,
        227,
        234,
        264,
        302,
        293,
        259,
        229,
        203,
        229,
        242,
        233,
        267,
        269,
        270,
        315,
        364,
        347,
        312,
        274,
        237,
        278,
        284,
        277,
        317,
        313,
        318,
        374,
        413,
        405,
        355,
        306,
        271,
        306,
        315,
        301,
        356,
        348,
        355,
        422,
        465,
        467,
        404,
        347,
        305,
        336,
        340,
        318,
        362,
        348,
        363,
        435,
        491,
        505,
        404,
        359,
        310,
        337,
        360,
        342,
        406,
        396,
        420,
        472,
        548,
        559,
        463,
        407,
        362,
        405,
        417,
        391,
        419,
        461,
        472,
        535,
        622,
        606,
        508,
        461,
        390,
        432,
    ],
    dtype=np.float32,
)


TS_DATASETS_URL = {
    "air_passengers": {
        "url": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",  # noqa: E501
        "format": "csv",
        "freq": "MS",
    },
    "volatility": {
        "url": "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip",
        "format": "zip",
        "csv_inside": "oxfordmanrealizedvolatilityindices.csv",
    },
    "electricity": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip",
        "format": "zip",
        "freq": "15T",
        "csv_inside": "LD2011_2014.txt",
    },
    "traffic": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip",
        "format": "zip",
        "freq": "H",
        "csv_inside": "PEMS_train",
    },
    "favorita": {
        "url": "https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data",
        "format": "kaggle",
    },
    "m5": {
        "url": "https://www.kaggle.com/c/m5-forecasting-accuracy/data",
        "format": "kaggle",
    },
}


def download_and_extract(name: str) -> str:
    """Robust download utility using Keras get_file logic."""
    if name not in TS_DATASETS_URL:
        raise ValueError(f"Dataset {name} configuration not found.")

    config = TS_DATASETS_URL[name]
    cache_dir = os.path.join(TFTS_DATASETS_CACHE, name)
    os.makedirs(cache_dir, exist_ok=True)

    path = get_file(
        fname=name, origin=config["url"], cache_subdir=cache_dir, extract=(config.get("format", "zip") == "zip")
    )
    return path


def get_data(
    name: str = "sine", train_length: int = 24, predict_sequence_length: int = 8, test_size: float = 0.1, **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")
    if name == "sine":
        return get_sine(train_length, predict_sequence_length, test_size=test_size, **kwargs)

    elif name == "airpassengers":
        return get_air_passengers(train_length, predict_sequence_length, test_size=test_size)

    elif name == "ar":
        return get_ar_data(**kwargs)

    elif name == "volatility":
        return get_volatility_data()

    elif name == "electricity":
        return get_electricity_data()

    elif name == "traffic":
        return get_traffic_data()

    else:
        raise ValueError(
            f"unsupported data of {name} yet, try 'sine', 'airpassengers', 'ar', 'volatility', 'electricity', 'traffic'"
        )


def get_sine(
    train_sequence_length: int = 24,
    predict_sequence_length: int = 8,
    test_size: float = 0.2,
    n_examples: int = 100,
    seed: Optional[int] = None,
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
    if train_sequence_length < 1 or predict_sequence_length < 1:
        raise ValueError("sequence lengths must be positive")
    if n_examples < 1:
        raise ValueError("n_examples must be positive")
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    x: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for _ in range(n_examples):
        rand = rng.uniform(0.0, 2.0 * np.pi)
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
    logging.info(f"Load sine data {x_array.shape} {y_array.shape}")

    if test_size > 0:
        split_idx = int(n_examples * (1 - test_size))
        x_train = x_array[:split_idx]
        y_train = y_array[:split_idx]
        x_valid = x_array[split_idx:]
        y_valid = y_array[split_idx:]
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
    if train_sequence_length < 1 or predict_sequence_length < 1:
        raise ValueError("train_sequence_length and predict_sequence_length must be positive")
    if train_sequence_length + predict_sequence_length > len(AIR_PASSENGERS_VALUES):
        raise ValueError("Requested sequence lengths exceed the AirPassengers dataset length")

    # Keep this canonical small dataset in-package so examples and tests work offline.
    v = AIR_PASSENGERS_VALUES.reshape(-1, 1).copy()
    v = (v - np.min(v)) / (np.max(v) - np.min(v))  # MinMaxScaler

    window_count = len(v) - train_sequence_length - predict_sequence_length + 1
    x_array = np.stack([v[i : i + train_sequence_length] for i in range(window_count)])
    y_array = np.stack(
        [
            v[i + train_sequence_length : i + train_sequence_length + predict_sequence_length]
            for i in range(window_count)
        ]
    )
    logging.info(f"Load air passenger data {x_array.shape} {y_array.shape}")

    if test_size > 0:
        split_idx = int(len(x_array) * (1 - test_size))
        x_train = x_array[:split_idx]
        y_train = y_array[:split_idx]
        x_valid = x_array[split_idx:]
        y_valid = y_array[split_idx:]
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


def get_volatility_data() -> pd.DataFrame:
    data_dir = download_and_extract("volatility")
    csv_path = os.path.join(data_dir, TS_DATASETS_URL["volatility"]["csv_inside"])

    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime([str(s).split("+")[0] for s in df.index])
    df = df.reset_index().rename(columns={"index": "date"})

    # Feature engineering from reference
    df["log_vol"] = np.log(df["rv5_ss"].replace(0, np.nan))
    df["log_vol"] = df.groupby("Symbol")["log_vol"].ffill().bfill()

    # Mapping regions
    symbol_region_mapping = {".AEX": "EMEA", ".DJI": "AMER", ".HSI": "APAC", ".SPX": "AMER"}  # truncated for brevity
    df["region"] = df["Symbol"].map(symbol_region_mapping).fillna("Unknown")

    return df


def get_electricity_data() -> pd.DataFrame:
    data_dir = download_and_extract("electricity")
    csv_path = os.path.join(data_dir, TS_DATASETS_URL["electricity"]["csv_inside"])

    # Industrial datasets are often large; use specific separators
    df = pd.read_csv(csv_path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.resample("1H").mean().replace(0.0, np.nan)

    # Melt to long format (productive for TimeSeriesSequence)
    df = df.reset_index().melt(id_vars="index", var_name="id", value_name="power_usage")
    df = df.rename(columns={"index": "date"}).dropna()
    return df


def get_traffic_data() -> pd.DataFrame:
    data_dir = download_and_extract("traffic")
    logger.info("Reading PEMS metadata files...")

    def _process_pems_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            l = [variable_type(i) for i in s.replace("[", "").replace("]", "").split()]
        else:
            l = [variable_type(i) for i in s.replace("[", "").replace("]", "").split(delimiter)]
        return l

    def _read_pems_matrix(data_folder, filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r") as dat:
            lines = dat.readlines()
            for i, line in enumerate(lines):
                # array is a list of lists (stations x time_observations)
                array = [
                    _process_pems_list(row_split, variable_type=float, delimiter=None)
                    for row_split in _process_pems_list(line, variable_type=str, delimiter=";")
                ]
                array_list.append(array)
        return array_list

    def read_single_list(fname):
        with open(os.path.join(data_dir, fname), "r") as f:
            return _process_pems_list(f.readlines()[0])

    shuffle_order = np.array(read_single_list("randperm")) - 1
    train_dayofweek = read_single_list("PEMS_trainlabels")
    test_dayofweek = read_single_list("PEMS_testlabels")
    stations_list = read_single_list("stations_list")

    logger.info("Reading and parsing train/test matrices (this may take a moment)...")
    train_tensor = _read_pems_matrix(data_dir, "PEMS_train")
    test_tensor = _read_pems_matrix(data_dir, "PEMS_test")

    # Inverse permutate shuffle order to restore temporal consistency
    inverse_mapping = {new_loc: prev_loc for prev_loc, new_loc in enumerate(shuffle_order)}
    reverse_shuffle_order = np.array([inverse_mapping[i] for i in range(len(shuffle_order))])

    # Combine and Reorder
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    logger.info("Aggregating to hourly data and formatting...")
    labels = [f"traj_{i}" for i in stations_list]
    hourly_list = []

    for day, day_matrix in enumerate(combined_tensor):
        # day_matrix.T -> index: time (144 intervals), columns: stations
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        # Sampled at 10 min intervals: 6 intervals = 1 hour
        hourly["hour_on_day"] = [int(i / 6) for i in hourly.index]

        # Mean occupancy per hour
        hourly = hourly.groupby("hour_on_day").mean()
        hourly["sensor_day"] = day
        hourly["time_on_day"] = hourly.index
        hourly["day_of_week"] = day_of_week[day]
        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True)

    # Flatten the dataframe: Each row is (sensor_id, time, occupancy)
    store_columns = [c for c in hourly_frame.columns if "traj" in c]
    other_columns = ["sensor_day", "time_on_day", "day_of_week"]

    flat_list = []
    for store in store_columns:
        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ["occupancy"] + other_columns
        sliced["station_id"] = int(store.replace("traj_", ""))

        # Calculate hours from start for a continuous time axis
        sliced["hours_from_start"] = sliced["time_on_day"] + sliced["sensor_day"] * 24.0
        flat_list.append(sliced)

    df = pd.concat(flat_list, axis=0, ignore_index=True)

    # Filter to match range used by academic papers (first 173 days)
    df = df[df["sensor_day"] < 173].copy()

    # Sorting for time-series consistency
    df = df.sort_values(["station_id", "hours_from_start"])

    return df
