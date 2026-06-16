"""Datasets migrated from the legacy ``examples/benchmarks`` folder."""

import logging
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from benchmark.base import Dataset
from benchmark.datasets.base import split_train_valid

logger = logging.getLogger(__name__)


def _resolve_data_path(value: str, env_name: str) -> str:
    path = value or os.environ.get(env_name, "")
    if path and os.path.isdir(path):
        return os.path.join(path, "train.csv")
    return path


def _window_arrays(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    feature_columns: Sequence[str],
    target_column: str,
    time_column: str,
    train_length: int,
    predict_sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    group_key = list(group_columns) if len(group_columns) > 1 else group_columns[0]

    for _, group in df.groupby(group_key, sort=False):
        group = group.sort_values(time_column)
        features = group[list(feature_columns)].to_numpy(dtype=np.float32)
        target = group[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
        limit = len(group) - train_length - predict_sequence_length + 1
        for start in range(max(0, limit)):
            x_list.append(features[start : start + train_length])
            y_list.append(target[start + train_length : start + train_length + predict_sequence_length])

    if not x_list:
        raise ValueError(
            "No benchmark windows were generated. "
            "Reduce train_length/predict_sequence_length or provide a longer dataset."
        )
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


class ForecastingStickerSalesDataset(Dataset):
    """Kaggle Playground sticker sales forecasting benchmark.

    This replaces ``examples/benchmarks/forecasting_sticker_sales`` with a
    standard benchmark dataset. Provide ``data_path`` as a CSV file or directory
    containing ``train.csv``. Expected Kaggle columns are ``date``, ``country``,
    ``store``, ``product`` and ``num_sold``.
    """

    name = "forecasting_sticker_sales"
    description = "Kaggle Playground sticker sales forecasting"
    train_length = 144
    predict_sequence_length = 32
    num_features = 4
    is_grouped = True
    target_column = "num_sold"
    time_column = "date"
    group_column = ["country", "store", "product"]

    def _load_dataframe(self, data_path: str = "", **kwargs) -> pd.DataFrame:
        path = _resolve_data_path(data_path, "STICKER_SALES_PATH")
        if path and os.path.exists(path):
            return pd.read_csv(path)

        logger.warning("Sticker sales data not found. Using deterministic synthetic placeholder data.")
        rng = np.random.default_rng(kwargs.get("seed", 42))
        dates = pd.date_range("2017-01-01", periods=240, freq="D")
        rows = []
        for country in ["Canada", "Finland", "Kenya"]:
            for store in ["KaggleMart", "KaggleRama"]:
                for product in ["Sticker A", "Sticker B"]:
                    base = 120 + 20 * (country == "Canada") + 12 * (store == "KaggleRama")
                    product_shift = 15 * (product == "Sticker B")
                    seasonal = 18 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
                    trend = np.linspace(0, 30, len(dates))
                    noise = rng.normal(0, 4, len(dates))
                    values = np.maximum(1, base + product_shift + seasonal + trend + noise)
                    for date, value in zip(dates, values):
                        rows.append(
                            {
                                "date": date,
                                "country": country,
                                "store": store,
                                "product": product,
                                "num_sold": value,
                            }
                        )
        return pd.DataFrame(rows)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["num_sold"] = pd.to_numeric(df["num_sold"], errors="coerce")
        df = df.dropna(subset=["date", "num_sold"])

        group_cols = ["country", "store", "product"]
        for column in group_cols:
            if column not in df:
                df[column] = "series"

        dayofweek = df["date"].dt.dayofweek.astype(np.float32)
        dayofyear = df["date"].dt.dayofyear.astype(np.float32)
        df["target_scaled"] = df.groupby(group_cols)["num_sold"].transform(
            lambda values: (values - values.mean()) / (values.std(ddof=0) + 1e-6)
        )
        df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
        df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
        return df

    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        train_length = kwargs.get("train_length") or self.train_length
        predict_length = kwargs.get("predict_sequence_length") or self.predict_sequence_length
        df = self._preprocess(self._load_dataframe(**kwargs))
        return _window_arrays(
            df=df,
            group_columns=["country", "store", "product"],
            feature_columns=["target_scaled", "dow_sin", "dow_cos", "doy_sin"],
            target_column="target_scaled",
            time_column="date",
            train_length=train_length,
            predict_sequence_length=predict_length,
        )

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x, y = self.prepare_data(**kwargs)
        return split_train_valid(x, y, test_size=kwargs.get("test_size", 0.2))


class CMIDetectSleepStatesDataset(Dataset):
    """CMI detect sleep states sequence benchmark.

    The legacy folder only provided a benchmark slot. This adapter supports
    common CMI-style CSV columns (``series_id``, ``step``/``timestamp``,
    ``anglez``, ``enmo`` and ``awake``/``target``) and falls back to synthetic
    sleep-state sequences when no local data is supplied.
    """

    name = "CMI_detect_sleep_states"
    description = "Child Mind Institute detect sleep states sequence benchmark"
    train_length = 144
    predict_sequence_length = 32
    num_features = 3
    is_grouped = True
    target_column = "awake"
    time_column = "step"
    group_column = "series_id"

    def _load_dataframe(self, data_path: str = "", **kwargs) -> pd.DataFrame:
        path = _resolve_data_path(data_path, "CMI_SLEEP_STATES_PATH")
        if path and os.path.exists(path):
            return pd.read_csv(path)

        logger.warning("CMI sleep states data not found. Using deterministic synthetic placeholder data.")
        rng = np.random.default_rng(kwargs.get("seed", 42))
        rows = []
        steps = np.arange(360)
        for series_idx in range(8):
            phase = series_idx * 0.35
            circadian = np.sin(2 * np.pi * steps / 96 + phase)
            awake = (circadian > -0.15).astype(np.float32)
            enmo = np.maximum(0, 0.04 + 0.18 * awake + rng.normal(0, 0.015, len(steps)))
            anglez = 20 * np.sin(2 * np.pi * steps / 48 + phase) + rng.normal(0, 3, len(steps))
            for step, angle, motion, state in zip(steps, anglez, enmo, awake):
                rows.append(
                    {
                        "series_id": f"series_{series_idx}",
                        "step": step,
                        "anglez": angle,
                        "enmo": motion,
                        "awake": state,
                    }
                )
        return pd.DataFrame(rows)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "series_id" not in df:
            df["series_id"] = "series"
        if "step" not in df:
            if "timestamp" in df:
                df["step"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
            else:
                df["step"] = df.groupby("series_id").cumcount()

        target_column = "awake" if "awake" in df else "target"
        if target_column not in df:
            raise ValueError("CMI sleep states data must include an 'awake' or 'target' column.")

        for column in ["anglez", "enmo"]:
            if column not in df:
                df[column] = 0.0

        df["awake"] = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(np.float32)
        df["anglez"] = pd.to_numeric(df["anglez"], errors="coerce").fillna(0).astype(np.float32) / 90.0
        df["enmo"] = pd.to_numeric(df["enmo"], errors="coerce").fillna(0).astype(np.float32)
        return df

    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        train_length = kwargs.get("train_length") or self.train_length
        predict_length = kwargs.get("predict_sequence_length") or self.predict_sequence_length
        df = self._preprocess(self._load_dataframe(**kwargs))
        return _window_arrays(
            df=df,
            group_columns=["series_id"],
            feature_columns=["anglez", "enmo", "awake"],
            target_column="awake",
            time_column="step",
            train_length=train_length,
            predict_sequence_length=predict_length,
        )

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x, y = self.prepare_data(**kwargs)
        return split_train_valid(x, y, test_size=kwargs.get("test_size", 0.2))
