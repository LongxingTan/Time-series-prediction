"""Reusable Stallion preprocessing for Temporal Fusion Transformers.

The module deliberately depends only on pandas/numpy.  Callers supply the raw
dataframe, so the preprocessing remains usable without making PyTorch a TFTS dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SPECIAL_DAYS = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
STATIC_CATEGORICALS = ["agency", "sku"]
STATIC_REALS = ["avg_population_2017", "avg_yearly_household_income_2017"]
KNOWN_CATEGORICALS = ["month", "special_days"]
KNOWN_REALS = ["time_idx", "price_regular", "discount_in_percent"]
UNKNOWN_REALS = [
    "volume",
    "log_volume",
    "industry_volume",
    "soda_volume",
    "avg_max_temp",
    "avg_volume_by_agency",
    "avg_volume_by_sku",
]
TEMPORAL_CATEGORICAL_COLUMNS = ["month"] + SPECIAL_DAYS


@dataclass
class StallionBatch:
    """Numpy tensors emitted by :class:`StallionPreprocessor`."""

    inputs: Dict[str, np.ndarray]
    target: np.ndarray
    target_original: np.ndarray
    group_keys: np.ndarray
    decoder_time_idx: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def feature_engineer_stallion(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()
    data["month"] = data.date.dt.month.astype(str).astype("category")
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
    data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")
    data[SPECIAL_DAYS] = data[SPECIAL_DAYS].apply(lambda column: column.map({0: "-", 1: column.name}))
    data[SPECIAL_DAYS] = data[SPECIAL_DAYS].astype("category")
    return data


class StallionPreprocessor:
    """Fit encoders/scalers and construct fixed 24-to-6 TFT windows.

    The target transform is its ``GroupNormalizer(..., transformation=
    "softplus")``: inverse-softplus followed by a per-(agency, sku) standard
    transform. Other real covariates use training-row standardization, matching
    ``TimeSeriesDataSet``'s default real-valued encoders.
    """

    min_encoder_length = 12
    max_encoder_length = 24
    min_prediction_length = 1
    max_prediction_length = 6

    def __init__(self):
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.real_scalers: Dict[str, Tuple[float, float]] = {}
        self.target_scales: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.training_cutoff: Optional[int] = None
        self.fitted = False

    @staticmethod
    def _softplus_inverse(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        eps = np.finfo(np.float64).eps
        return np.where(
            values > 20.0,
            values,
            values + np.log(-np.expm1(-(values + eps))),
        )

    def fit(self, data: pd.DataFrame) -> "StallionPreprocessor":
        if "time_idx" not in data or "log_volume" not in data:
            data = feature_engineer_stallion(data)
        self.training_cutoff = int(data.time_idx.max() - self.max_prediction_length)
        training = data[data.time_idx <= self.training_cutoff]

        for column in STATIC_CATEGORICALS + TEMPORAL_CATEGORICAL_COLUMNS:
            values = sorted(training[column].astype(str).unique().tolist())
            self.category_maps[column] = {value: index + 1 for index, value in enumerate(values)}

        # Volume is normalized per group below, not by a global real scaler.
        for column in STATIC_REALS + KNOWN_REALS + UNKNOWN_REALS[1:]:
            values = training[column].to_numpy(dtype=np.float64)
            center = float(np.mean(values))
            scale = float(np.std(values, ddof=0))
            self.real_scalers[column] = (center, scale if scale > 1e-12 else 1.0)

        transformed = self._softplus_inverse(training.volume.to_numpy())
        scale_frame = training[["agency", "sku"]].copy()
        scale_frame["transformed_target"] = transformed
        grouped = scale_frame.groupby(["agency", "sku"], observed=True).transformed_target
        eps = np.finfo(np.float16).eps
        for key, series in grouped:
            # pandas std(ddof=1) is what GroupNormalizer uses for grouped data.
            self.target_scales[(str(key[0]), str(key[1]))] = (
                float(series.mean()),
                float(series.std(ddof=1) + eps),
            )
        self.fitted = True
        return self

    def _encode_category(self, series: pd.Series, column: str) -> np.ndarray:
        mapping = self.category_maps[column]
        return series.astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int32)

    def _scale_real(self, series: pd.Series, column: str) -> np.ndarray:
        center, scale = self.real_scalers[column]
        return ((series.to_numpy(dtype=np.float64) - center) / scale).astype(np.float32)

    def _target_parameters(self, frame: pd.DataFrame) -> Tuple[float, float]:
        key = (str(frame.agency.iloc[0]), str(frame.sku.iloc[0]))
        return self.target_scales[key]

    def _target_transform(self, values: np.ndarray, center: float, scale: float) -> np.ndarray:
        return ((self._softplus_inverse(values) - center) / scale).astype(np.float32)

    def inverse_target(self, values: np.ndarray, target_scale: np.ndarray) -> np.ndarray:
        """Return normalized predictions to original volume units."""
        normalized = np.asarray(values, dtype=np.float64)
        params = np.asarray(target_scale, dtype=np.float64)
        while params.ndim < normalized.ndim + 1:
            params = np.expand_dims(params, axis=-2)
        transformed = normalized * params[..., 1] + params[..., 0]
        return np.logaddexp(0.0, transformed).astype(np.float32)

    def transform(self, data: pd.DataFrame, split: str) -> StallionBatch:
        """Build training sliding windows or the last-six-month validation windows."""
        if not self.fitted:
            raise RuntimeError("Call fit(data) before transform(data, split)")
        if "time_idx" not in data or "log_volume" not in data:
            data = feature_engineer_stallion(data)
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")

        records = []
        for _, group in data.groupby(["agency", "sku"], observed=True, sort=True):
            group = group.sort_values("time_idx").reset_index(drop=True)
            if split == "validation":
                decoder_starts = [int(np.searchsorted(group.time_idx.to_numpy(), self.training_cutoff + 1))]
            else:
                available = group.index[group.time_idx <= self.training_cutoff].to_numpy()
                decoder_starts = [
                    int(index)
                    for index in available
                    if index >= self.max_encoder_length and index + self.max_prediction_length - 1 <= available[-1]
                ]
            for decoder_start in decoder_starts:
                encoder = group.iloc[decoder_start - self.max_encoder_length : decoder_start]
                decoder = group.iloc[decoder_start : decoder_start + self.max_prediction_length]
                if len(encoder) == self.max_encoder_length and len(decoder) == self.max_prediction_length:
                    records.append(self._window(encoder, decoder))
        if not records:
            raise ValueError(f"No {split} windows could be constructed")
        keys = records[0][0].keys()
        inputs = {key: np.stack([record[0][key] for record in records]) for key in keys}
        return StallionBatch(
            inputs=inputs,
            target=np.stack([record[1] for record in records]),
            target_original=np.stack([record[2] for record in records]),
            group_keys=np.asarray([record[3] for record in records], dtype=str),
            decoder_time_idx=np.stack([record[4] for record in records]),
        )

    def _window(self, encoder: pd.DataFrame, decoder: pd.DataFrame):
        whole = pd.concat([encoder, decoder], ignore_index=True)
        center, scale = self._target_parameters(encoder)

        static_cat = np.asarray(
            [self._encode_category(encoder.iloc[:1][column], column)[0] for column in STATIC_CATEGORICALS],
            dtype=np.int32,
        )
        static_real = np.asarray(
            [self._scale_real(encoder.iloc[:1][column], column)[0] for column in STATIC_REALS],
            dtype=np.float32,
        )
        temporal_cat = np.stack(
            [self._encode_category(whole[column], column) for column in TEMPORAL_CATEGORICAL_COLUMNS],
            axis=-1,
        )
        known_real = np.stack([self._scale_real(whole[column], column) for column in KNOWN_REALS], axis=-1)
        unknown_real_parts = [self._target_transform(encoder.volume.to_numpy(), center, scale)] + [
            self._scale_real(encoder[column], column) for column in UNKNOWN_REALS[1:]
        ]
        encoder_real = np.concatenate(
            [known_real[: self.max_encoder_length], np.stack(unknown_real_parts, axis=-1)],
            axis=-1,
        )
        original_target = decoder.volume.to_numpy(dtype=np.float32)[:, None]
        normalized_target = self._target_transform(decoder.volume.to_numpy(), center, scale)[:, None]
        return (
            {
                "static_categorical": static_cat,
                "static_real": static_real,
                "encoder_categorical": temporal_cat[: self.max_encoder_length],
                "encoder_real": encoder_real,
                "decoder_categorical": temporal_cat[self.max_encoder_length :],
                "decoder_real": known_real[self.max_encoder_length :],
                "target_scale": np.asarray([center, scale], dtype=np.float32),
            },
            normalized_target,
            original_target,
            [str(encoder.agency.iloc[0]), str(encoder.sku.iloc[0])],
            decoder.time_idx.to_numpy(dtype=np.int32),
        )

    @property
    def categorical_cardinalities(self) -> Mapping[str, List[int]]:
        return {
            "static": [len(self.category_maps[column]) + 1 for column in STATIC_CATEGORICALS],
            "temporal": [len(self.category_maps[column]) + 1 for column in TEMPORAL_CATEGORICAL_COLUMNS],
        }

    def metadata(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Call fit(data) before metadata()")
        return {
            "training_cutoff": self.training_cutoff,
            "min_encoder_length": self.min_encoder_length,
            "max_encoder_length": self.max_encoder_length,
            "min_prediction_length": self.min_prediction_length,
            "max_prediction_length": self.max_prediction_length,
            "static_categoricals": STATIC_CATEGORICALS,
            "static_reals": STATIC_REALS,
            "time_varying_known_categoricals": KNOWN_CATEGORICALS,
            "special_days_members": SPECIAL_DAYS,
            "time_varying_known_reals": KNOWN_REALS,
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": UNKNOWN_REALS,
            "temporal_categorical_columns": TEMPORAL_CATEGORICAL_COLUMNS,
            "categorical_cardinalities": self.categorical_cardinalities,
            "target_normalizer": {
                "class": "GroupNormalizer-compatible",
                "groups": ["agency", "sku"],
                "method": "standard",
                "transformation": "softplus",
                "std_ddof": 1,
                "epsilon": float(np.finfo(np.float16).eps),
            },
        }
