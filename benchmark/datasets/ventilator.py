"""Breath-level ventilator data and the reference notebook's evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # noqa: PANDAS_OK - reuse the project's declared data dependency

from benchmark.base import Dataset


class VentilatorDataError(ValueError):
    """The supplied CSV cannot represent complete ventilator breaths."""


def engineer_reference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce cdeotte's ``add_features`` transforms on the raw CSV frame.

    Returns a feature frame whose rows are still ordered by the original CSV
    rows (breath-major, then time). Lags/leads use the existing row order, so
    callers must sort by ``breath_id`` then ``time_step`` first. One-hot R, C
    and the ``R__C`` product expand the width; all columns are returned.
    """
    df = frame.copy()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby("breath_id")["time_step"].cumsum()
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    for n in (1, 2, 3, 4):
        df[f"u_in_lag{n}"] = df.groupby("breath_id")["u_in"].shift(n)
        df[f"u_out_lag{n}"] = df.groupby("breath_id")["u_out"].shift(n)
        df[f"u_in_lag_back{n}"] = df.groupby("breath_id")["u_in"].shift(-n)
        df[f"u_out_lag_back{n}"] = df.groupby("breath_id")["u_out"].shift(-n)
    df = df.fillna(0)
    df["breath_id__u_in__max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby("breath_id")["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = df["breath_id__u_in__max"] - df["u_in"]
    df["breath_id__u_in__diffmean"] = df["breath_id__u_in__mean"] - df["u_in"]
    for n in (1, 2, 3, 4):
        df[f"u_in_diff{n}"] = df["u_in"] - df[f"u_in_lag{n}"]
        df[f"u_out_diff{n}"] = df["u_out"] - df[f"u_out_lag{n}"]
    df["one"] = 1
    df["count"] = df.groupby("breath_id")["one"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]
    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select([df["breath_id_lag"] == df["breath_id"]], [1], 0)
    df["breath_id_lag2same"] = np.select([df["breath_id_lag2"] == df["breath_id"]], [1], 0)
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = df.groupby("breath_id")["u_in"].ewm(halflife=9).mean().reset_index(level=0, drop=True)
    rolls = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg({"15_in_sum": "sum", "15_in_min": "min", "15_in_max": "max", "15_in_mean": "mean"})
        .reset_index(level=0, drop=True)
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = rolls.to_numpy()
    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    return pd.get_dummies(df)


class VentilatorDataset(Dataset):
    """Use complete breaths as samples; pressure is never an input feature.

    State holds the loaded arrays so every model can reuse identical folds.
    Fold assignment reproduces unshuffled GroupKFold for equal-sized groups,
    including its reverse argsort tie ordering. Scoring excludes expiration.
    """

    name = "ventilator"
    description = "Google Brain ventilator pressure, five-fold breath-group validation"
    train_length = 80
    predict_sequence_length = 80
    num_features = 5
    is_multivariate = True
    is_grouped = True
    target_column = "pressure"
    time_column = "time_step"
    group_column = "breath_id"

    def prepare_data(self, **kwargs: str | int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Load and verify full breaths sorted by ID and time."""
        path = Path(str(kwargs.get("path", "data/ventilator/train.csv")))
        frame = pd.read_csv(path).sort_values(["breath_id", "time_step"])
        counts = frame.groupby("breath_id", sort=True).size()
        if len(counts) < 5 or not np.all(counts.to_numpy() == 80):
            raise VentilatorDataError("Expected at least five complete 80-row breaths")
        columns = ["R", "C", "u_in", "u_out", "pressure", "time_step"]
        values = frame[columns].to_numpy(dtype=np.float32).reshape(-1, 80, len(columns))
        if not np.isfinite(values).all():
            raise VentilatorDataError("All numeric inputs and targets must be finite")
        if not np.isin(values[..., 3], [0, 1]).all():
            raise VentilatorDataError("u_out must be binary")
        self.breath_ids = counts.index.to_numpy(dtype=np.int64)
        self.mask = 1 - values[..., 3:4]
        if np.any(self.mask.sum(axis=1) == 0):
            raise VentilatorDataError("Every breath must include inspiration")
        self.x = np.stack(
            [values[..., 0], values[..., 1], values[..., 2], np.cumsum(values[..., 2], axis=1), values[..., 3]], axis=-1
        )
        self.y = values[..., 4:5]
        return self.x, self.y

    def prepare_reference(
        self, path: str | Path = "data/ventilator/train.csv", force_float: bool = True
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Return the engineered, robust-scaled features and multi-channel targets.

        Reproduces cdeotte's transformer notebook inputs: feature engineering,
        manual RobustScaler (median center, inter-quartile scale) fit across the
        whole frame, plus a three-channel target stack ``[pressure,
        pressure_diff, pressure_integral]`` and the inspiration mask. The fold
        split and mask both index the same breath order as ``prepare_data``.
        """
        frame = pd.read_csv(path).sort_values(["breath_id", "time_step"])
        counts = frame.groupby("breath_id", sort=True).size()
        if len(counts) < 5 or not np.all(counts.to_numpy() == 80):
            raise VentilatorDataError("Expected at least five complete 80-row breaths")
        features = engineer_reference_features(frame)
        if force_float:
            features = features.astype(np.float32)
        features["pressure_diff"] = features.groupby("breath_id")["pressure"].diff().fillna(0)
        features["pressure_integral"] = features.groupby("breath_id")["pressure"].cumsum() / 200
        drop = [
            "pressure",
            "pressure_diff",
            "pressure_integral",
            "id",
            "breath_id",
            "one",
            "count",
            "breath_id_lag",
            "breath_id_lag2",
            "breath_id_lagsame",
            "breath_id_lag2same",
        ]
        targets = (
            features[["pressure", "pressure_diff", "pressure_integral"]].to_numpy(dtype=np.float32).reshape(-1, 80, 3)
        )
        matrix = features.drop(columns=drop)
        median = matrix.median(axis=0).to_numpy(dtype=np.float32)
        lower = matrix.quantile(0.25, axis=0).to_numpy(dtype=np.float32)
        upper = matrix.quantile(0.75, axis=0).to_numpy(dtype=np.float32)
        scale = upper - lower
        scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0
        scaled = (
            ((matrix.to_numpy(dtype=np.float32) - median) / scale).reshape(-1, 80, matrix.shape[-1]).astype(np.float32)
        )
        u_out = frame["u_out"].to_numpy(dtype=np.float32).reshape(-1, 80, 1)
        self.breath_ids = counts.index.to_numpy(dtype=np.int64)
        self.mask = 1 - u_out
        if not np.isfinite(scaled).all() or not np.isfinite(targets).all():
            raise VentilatorDataError("Engineered inputs and targets must be finite")
        return scaled, targets, self.mask

    def fold_indices(self, fold: int = 0) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Reproduce sklearn GroupKFold(n_splits=5) on the CSV rows."""
        if fold not in range(5):
            raise VentilatorDataError("fold must be between 0 and 4")
        order = np.full(len(self.breath_ids), 80, dtype=np.int64).argsort()[::-1]
        loads = np.zeros(5, dtype=np.int64)
        assignments = np.empty(len(order), dtype=np.int64)
        for index in order:
            selected = int(loads.argmin())
            assignments[index] = selected
            loads[selected] += 80
        return np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)

    def get_train_valid_split(self, **kwargs: str | int):
        """Return the selected fold without splitting any breath."""
        self.prepare_data(**kwargs)
        train, valid = self.fold_indices(int(kwargs.get("fold", 0)))
        return (self.x[train], self.y[train]), (self.x[valid], self.y[valid])

    def load_folds(self, path: Path) -> list[tuple[int, NDArray[np.int64]]]:
        """Recover saved validation indices and verify their breath identity."""
        if not path.exists():
            return [(fold, self.fold_indices(fold)[1]) for fold in range(5)]
        with np.load(path) as archive:
            if not np.array_equal(archive["breath_ids"], self.breath_ids):
                raise VentilatorDataError("Saved folds do not match the loaded breath IDs")
            folds = sorted(
                (int(key.split("_", 1)[1]), archive[key]) for key in archive.files if key.startswith("valid_")
            )
        if [fold for fold, _ in folds] != list(range(5)):
            raise VentilatorDataError("A complete run requires all five saved folds")
        for _, valid in folds:
            if valid.ndim != 1 or not np.issubdtype(valid.dtype, np.integer) or len(valid) == 0:
                raise VentilatorDataError("Fold indices must be nonempty integer vectors")
        if not np.array_equal(np.sort(np.concatenate([valid for _, valid in folds])), np.arange(len(self.breath_ids))):
            raise VentilatorDataError("Saved folds must partition every breath exactly once")
        return folds


def inspiration_mae(target: NDArray[np.float32], prediction: NDArray[np.float32], mask: NDArray[np.float32]) -> float:
    """Compute the global masked MAE, not an average of per-breath MAEs."""
    if target.shape != prediction.shape or target.shape != mask.shape:
        raise VentilatorDataError("Target, prediction and inspiration mask shapes must match")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise VentilatorDataError("Scoring requires finite targets and predictions")
    if not np.isin(mask, [0, 1]).all() or mask.sum() == 0:
        raise VentilatorDataError("Scoring requires a nonempty binary inspiration mask")
    return float(np.sum(np.abs(target - prediction) * mask, dtype=np.float64) / mask.sum())
