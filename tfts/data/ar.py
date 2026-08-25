"""AR (autoregressive univariate) data preprocessing for N-BEATS parity.

synthetic AR dataset generation and its train/validation windowing

N-BEATS is target-only. In line with ``tfts``'s ``(x, encoder_feature,
decoder_feature)`` convention the pipeline emits ``x`` as the lookback window
of ``value`` and leaves both feature tensors empty (no covariates / static
slots exist for this model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "generate_ar_data",
    "feature_engineer_ar",
    "ARNBeatsPreprocessor",
    "ARDeepARPreprocessor",
]


def generate_ar_data(
    n_series: int = 100,
    timesteps: int = 400,
    seasonality: float = 10.0,
    trend: float = 3.0,
    noise: float = 0.1,
    level: float = 1.0,
    exp: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the identical synthetic AR dataset"""
    np.random.seed(seed)
    linear_trends = np.random.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = np.random.normal(size=n_series)[:, None] / timesteps**2
    seasonalities = np.random.normal(size=n_series)[:, None]
    levels = level * np.random.normal(size=n_series)[:, None]

    # generate series
    x = np.arange(timesteps)[None, :]
    series = (x * linear_trends + x**2 * quadratic_trends) * trend + seasonalities * np.sin(
        2 * np.pi * seasonality * x / timesteps
    )
    # add noise
    series = levels * series * (1 + noise * np.random.normal(size=series.shape))
    if exp:
        series = np.exp(series)

    # insert into dataframe, matching the PF column layout
    data = (
        pd.DataFrame(series)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "series", "level_1": "time_idx", 0: "value"})
    )
    return data


def feature_engineer_ar(data: pd.DataFrame) -> pd.DataFrame:
    """Simple pass-through (no feature engineering, matching Phase 1 spec)."""
    return data.copy()


@dataclass
class ARBatch:
    x: np.ndarray  # (n, encoder_length, 1)
    encoder_feature: Optional[np.ndarray] = None
    decoder_feature: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None  # (n, prediction_length, 1)
    target_original: Optional[np.ndarray] = None
    series: Optional[np.ndarray] = None
    decoder_time_idx: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


class ARNBeatsPreprocessor:
    """Build target-only N-BEATS windows from ``generate_ar_data`` output.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain ``series``, ``time_idx`` and ``value`` columns.
    encoder_length : int
        fixed lookback window length (matches PF ``max_encoder_length``).
    prediction_length : int
        fixed forecast horizon (matches PF ``max_prediction_length``).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        encoder_length: int = 60,
        prediction_length: int = 20,
    ) -> None:
        missing = {"series", "time_idx", "value"} - set(data.columns)
        if missing:
            raise ValueError(f"AR data missing required columns: {sorted(missing)}")
        self.data = data
        self.encoder_length = int(encoder_length)
        self.prediction_length = int(prediction_length)
        self.training_cutoff = int(data["time_idx"].max() - self.prediction_length)
        self.series_ids = sorted(data["series"].unique().tolist())
        self.n_series = len(self.series_ids)
        self._series_values = {
            sid: data.loc[data["series"] == sid, "value"].to_numpy(dtype=np.float32) for sid in self.series_ids
        }

    # ------------------------------------------------------------------ helpers
    def _windows_for(self, series_vals: np.ndarray, starts: List[int]):
        x = np.stack([series_vals[s : s + self.encoder_length] for s in starts])
        y = np.stack(
            [series_vals[s + self.encoder_length : s + self.encoder_length + self.prediction_length] for s in starts]
        )
        return x[:, :, None], y[:, :, None]

    # ------------------------------------------------------------------ splits
    def train(self) -> ARBatch:
        """Sliding windows fully contained in ``time_idx <= training_cutoff``."""
        # last train start keeps prediction window end <= cutoff
        last_start = self.training_cutoff - self.encoder_length - self.prediction_length + 1
        starts = list(range(last_start + 1))
        xs, ys, sids = [], [], []
        for sid in self.series_ids:
            x, y = self._windows_for(self._series_values[sid], starts)
            xs.append(x)
            ys.append(y)
            sids.append(np.full(len(starts), sid, dtype=np.int64))
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        sids = np.concatenate(sids, axis=0)
        return ARBatch(
            x=xs,
            y=ys,
            target_original=ys,
            series=sids,
            metadata={
                "mode": "train",
                "training_cutoff": self.training_cutoff,
                "n_samples": int(xs.shape[0]),
            },
        )

    def validation(self) -> ARBatch:
        """One forecast per series starting at ``training_cutoff + 1``."""
        start = self.training_cutoff + 1 - self.encoder_length  # encoder [cutoff+1-60, cutoff+1)
        xs, ys, sids, dec_idx = [], [], [], []
        for sid in self.series_ids:
            v = self._series_values[sid]
            x = v[start : start + self.encoder_length][None, :, None]
            y = v[start + self.encoder_length : start + self.encoder_length + self.prediction_length][None, :, None]
            xs.append(x)
            ys.append(y)
            sids.append(sid)
            dec_idx.append(np.arange(self.training_cutoff + 1, self.training_cutoff + 1 + self.prediction_length))
        return ARBatch(
            x=np.concatenate(xs, axis=0),
            y=np.concatenate(ys, axis=0),
            target_original=np.concatenate(ys, axis=0),
            series=np.asarray(sids, dtype=np.int64),
            decoder_time_idx=np.asarray(dec_idx, dtype=np.int64),
            metadata={
                "mode": "validation",
                "training_cutoff": self.training_cutoff,
                "n_samples": len(xs),
            },
        )

    def metadata(self) -> Dict:
        return {
            "encoder_length": self.encoder_length,
            "prediction_length": self.prediction_length,
            "training_cutoff": self.training_cutoff,
            "series_count": self.n_series,
            "train_samples": self.train().metadata["n_samples"],
            "validation_samples": self.validation().metadata["n_samples"],
            "time_varying_unknown_reals": ["value"],
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "target_normalizer": {"class": "none", "note": "raw univariate scale, no normalization"},
            "generation": {"seasonality": 10.0, "timesteps": 400, "n_series": 100, "seed": 42},
        }


@dataclass
class ARDeepARBatch:
    x: np.ndarray  # (n, encoder_length, 1) NORMALIZED encoder values
    decoder_feature: np.ndarray  # (n, prediction_length, 1) NORMALIZED teacher-forced lagged target
    static: np.ndarray  # (n, 1) series id (int, for the embedding lookup)
    y: np.ndarray  # (n, prediction_length, 1) NORMALIZED target (for NLL loss)
    target_original: Optional[np.ndarray] = None  # (n, prediction_length, 1) raw target (for eval)
    decoder_time_idx: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


class ARDeepARPreprocessor:
    """DeepAR windows from ``generate_ar_data`` output, replicating the Phase 1 spec.

    Feature-for-feature the Forecasting pipeline:
      - identical ``generate_ar_data`` call/seed and train/validation split boundaries,
      - fixed encoder/decoder lengths (60 / 20),
      - ``value`` as the sole time-varying real, ``series`` as a static categorical
        (embedding lookup, one id per series),
      - target normalized with the Phase 1 ``EncoderNormalizer`` fit (global mean/std over
        the training target; ``transformation`` recorded by Phase 1, ``None`` for this data).

    DeepAR's decoding is autoregressive with **teacher forcing during training**, so
    ``decoder_feature`` is the *lagged* target: the previous time step's value, with the last
    encoder value as the first decoder input. Shape convention:
      ``decoder_feature[t] = [x[-1], y[0], y[1], ..., y[-2]][t]``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        encoder_length: int = 60,
        prediction_length: int = 20,
        mean: float = 0.0,
        std: float = 1.0,
        transformation: Optional[str] = None,
    ) -> None:
        missing = {"series", "time_idx", "value"} - set(data.columns)
        if missing:
            raise ValueError(f"AR data missing required columns: {sorted(missing)}")
        self.data = data
        self.encoder_length = int(encoder_length)
        self.prediction_length = int(prediction_length)
        self.training_cutoff = int(data["time_idx"].max() - self.prediction_length)
        self.series_ids = sorted(data["series"].unique().tolist())
        self.n_series = len(self.series_ids)
        self._series_values = {
            sid: data.loc[data["series"] == sid, "value"].to_numpy(dtype=np.float32) for sid in self.series_ids
        }
        self.mean = float(mean)
        self.std = float(std)
        self.transformation = transformation
        # encode series id -> dense 0..n-1 for the embedding lookup
        self._series_to_idx = {sid: i for i, sid in enumerate(self.series_ids)}
        # NaNLabelEncoder in PF maps the first encountered class to 0 as well; identity here.

    # ------------------------------------------------------------------ normalization
    def _forward_transform(self, v: np.ndarray) -> np.ndarray:
        """Transformation applied before standardizing (mirrors PF preprocess())."""
        t = self.transformation
        if t is None or t == "relu":
            return v
        if t == "log":
            return np.log(np.clip(v, 1e-7, None))
        if t == "softplus":
            return np.log(np.maximum(np.expm1(np.where(v > 20, 20.0, v)), 1e-7))
        if t == "count":
            return v + 1.0
        if t == "logit":
            return np.log(np.clip(v, 1e-7, 1 - 1e-7) / (1 - np.clip(v, 1e-7, 1 - 1e-7)))
        if callable(t):
            return np.asarray(t(v), dtype=np.float32)
        raise ValueError(f"unsupported transformation: {t!r}")

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return (self._forward_transform(v) - self.mean) / self.std

    def _inverse(self, v: np.ndarray) -> np.ndarray:
        """Inverse of normalize() (standardize back, then reverse transformation)."""
        z = v * self.std + self.mean
        t = self.transformation
        if t is None:
            return z
        if t == "relu":
            return np.maximum(z, 0.0)
        if t == "log":
            return np.exp(z)
        if t == "softplus":
            return np.log1p(np.exp(np.clip(z, -50, 50)))
        if t == "count":
            return np.maximum(z - 1.0, 0.0)
        if t == "logit":
            return 1.0 / (1.0 + np.exp(-z))
        if callable(t):
            raise NotImplementedError("callable transformation inverse not supported")
        raise ValueError(f"unsupported transformation: {t!r}")

    # ------------------------------------------------- public normalization API
    def transform(self, v: np.ndarray) -> np.ndarray:
        """Normalize raw values to model space (standardize after optional transform).

        Kept public so a forecasting pipeline can map raw inputs into generated-model
        space; ``inverse_transform`` reverses it.
        """
        return self._normalize(v)

    def inverse_transform(self, v: np.ndarray) -> np.ndarray:
        """Convert model-space (standardized) values back to original units."""
        return self._inverse(v)

    def normalizer_state(self) -> Dict:
        """Serializable normalizer state (independent of any trained weights)."""
        return {
            "class": "EncoderNormalizer (global fit, replicated from Phase 1)",
            "mean": float(self.mean),
            "std": float(self.std),
            "transformation": self.transformation,
        }

    # ------------------------------------------------------------------ helpers
    def _windows_for(self, series_vals: np.ndarray, starts: List[int]):
        """Return (normalized encoder, normalized teacher-forced decoder feat, norm y, raw y)."""
        xs, decs, ys, ys_orig = [], [], [], []
        for s in starts:
            enc = series_vals[s : s + self.encoder_length]  # v0..v59
            y = series_vals[s + self.encoder_length : s + self.encoder_length + self.prediction_length]  # w0..w19
            # teacher-forced lagged decoder input: [last encoder value, y[0..-2]]
            dec = np.concatenate([enc[-1:], y[:-1]])
            xs.append(self._normalize(enc))
            decs.append(self._normalize(dec))
            ys.append(self._normalize(y))
            ys_orig.append(y)
        return (
            np.stack(xs)[:, :, None],
            np.stack(decs)[:, :, None],
            np.stack(ys)[:, :, None],
            np.stack(ys_orig)[:, :, None],
        )

    # ------------------------------------------------------------------ splits
    def _batch(self, starts: List[int], sids: List[object], mode: str) -> ARDeepARBatch:
        xs, decs, ys, ys_orig = [], [], [], []
        static, dec_idx = [], []
        for sid in sids:
            v = self._series_values[sid]
            x, d, y, yo = self._windows_for(v, starts)
            xs.append(x)
            decs.append(d)
            ys.append(y)
            ys_orig.append(yo)
            static.append(np.full((len(starts), 1), self._series_to_idx[sid], dtype=np.int64))
            if mode == "validation":
                dec_idx.append(self.training_cutoff + 1 + np.arange(self.prediction_length))
        return ARDeepARBatch(
            x=np.concatenate(xs, axis=0),
            decoder_feature=np.concatenate(decs, axis=0),
            static=np.concatenate(static, axis=0),
            y=np.concatenate(ys, axis=0),
            target_original=np.concatenate(ys_orig, axis=0),
            decoder_time_idx=np.asarray(dec_idx, dtype=np.int64) if dec_idx else None,
            metadata={
                "mode": mode,
                "training_cutoff": self.training_cutoff,
                "n_samples": int(np.concatenate(xs, axis=0).shape[0]),
                "mean": self.mean,
                "std": self.std,
                "transformation": self.transformation,
            },
        )

    def train(self) -> ARDeepARBatch:
        """Sliding windows fully contained in ``time_idx <= training_cutoff``."""
        last_start = self.training_cutoff - self.encoder_length - self.prediction_length + 1
        starts = list(range(last_start + 1))
        return self._batch(starts, self.series_ids, mode="train")

    def validation(self) -> ARDeepARBatch:
        """One forecast per series starting at ``training_cutoff + 1``."""
        start = self.training_cutoff + 1 - self.encoder_length  # encoder [cutoff+1-60, cutoff+1)
        return self._batch([start], self.series_ids, mode="validation")

    def metadata(self) -> Dict:
        tr = self.train()
        va = self.validation()
        return {
            "encoder_length": self.encoder_length,
            "prediction_length": self.prediction_length,
            "training_cutoff": self.training_cutoff,
            "series_count": self.n_series,
            "train_samples": int(tr.metadata["n_samples"]),
            "validation_samples": int(va.metadata["n_samples"]),
            "time_varying_unknown_reals": ["value"],
            "static_categoricals": ["series"],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "target_normalizer": {
                "class": "EncoderNormalizer (global fit, replicated from Phase 1)",
                "transformation": self.transformation,
                "mean": float(self.mean),
                "std": float(self.std),
            },
            "decoder_input_convention": (
                "teacher-forced lagged target: [last_encoder_value, y[0], y[1], ..., y[-2]], " "normalized"
            ),
            "generation": {"seasonality": 10.0, "timesteps": 400, "n_series": 100, "seed": 42},
        }
