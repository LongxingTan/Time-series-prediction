"""M4 benchmark datasets backed by the archives used in ``exps``."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from benchmark.base import Dataset


class M4Dataset(Dataset):
    """Build deterministic rolling windows for one M4 seasonal pattern."""

    name = "m4"
    description = "M4 competition series loaded from local NPZ archives."
    train_length = 26
    predict_sequence_length = 13

    def prepare_data(self, **kwargs):
        return self.get_train_valid_split(**kwargs)[0]

    def get_train_valid_split(self, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        data_dir = Path(kwargs.get("data_dir", "exps/dataset/m4"))
        pattern = kwargs.get("seasonal_pattern", "Weekly")
        train_length = int(kwargs.get("train_length") or self.train_length)
        prediction_length = int(kwargs.get("predict_sequence_length") or self.predict_sequence_length)
        windows_per_series = int(kwargs.get("windows_per_series", 1))
        seed = int(kwargs.get("window_seed", kwargs.get("seed", 7)))

        info = pd.read_csv(data_dir / "M4-info.csv")
        histories = np.load(data_dir / "training.npz", allow_pickle=True)
        test_values = np.load(data_dir / "test.npz", allow_pickle=True)
        indices = np.flatnonzero(info["SP"].to_numpy() == pattern)
        if not len(indices):
            raise ValueError(f"No M4 series found for seasonal_pattern={pattern!r}")

        selected_histories = [self._clean(histories[index]) for index in indices]
        selected_targets = [self._clean(test_values[index]) for index in indices]
        x_train, y_train = self._training_windows(
            selected_histories,
            train_length,
            prediction_length,
            windows_per_series,
            np.random.default_rng(seed),
        )
        x_valid = np.stack([self._left_pad(values, train_length) for values in selected_histories])
        y_valid = np.stack([self._right_pad(values, prediction_length) for values in selected_targets])
        return (x_train[..., None], y_train[..., None]), (x_valid[..., None], y_valid[..., None])

    @staticmethod
    def _clean(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values[~np.isnan(values)]

    @staticmethod
    def _left_pad(values: np.ndarray, length: int) -> np.ndarray:
        output = np.zeros(length, dtype=np.float32)
        window = values[-length:]
        if len(window):
            output[-len(window) :] = window
        return output

    @staticmethod
    def _right_pad(values: np.ndarray, length: int) -> np.ndarray:
        output = np.zeros(length, dtype=np.float32)
        window = values[:length]
        if len(window):
            output[: len(window)] = window
        return output

    @classmethod
    def _training_windows(
        cls,
        histories: List[np.ndarray],
        train_length: int,
        prediction_length: int,
        windows_per_series: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_rows, y_rows = [], []
        for values in histories:
            latest_cutoff = len(values) - prediction_length
            if latest_cutoff < 1:
                continue
            earliest_cutoff = max(1, latest_cutoff - 10 * prediction_length)
            for _ in range(windows_per_series):
                cutoff = int(rng.integers(earliest_cutoff, latest_cutoff + 1))
                x_rows.append(cls._left_pad(values[:cutoff], train_length))
                y_rows.append(values[cutoff : cutoff + prediction_length])
        if not x_rows:
            raise ValueError("M4 selection contains no series long enough for a complete training window")
        return np.stack(x_rows), np.stack(y_rows)
