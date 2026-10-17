"""Verify that every saved checkpoint, reloaded and re-run, reproduces its metrics.

``benchmark.ventilator_bert`` trains the reference transformer and the TFTS BERT
on five folds, saves per-fold weights (``{model}_fold{fold}.weights.h5``) and an
out-of-fold prediction stack (``{model}_oof.npy``), and records one global
inspiration-MAE per model in ``results.json``. ``benchmark.ventilator`` does the
same for the full reference LSTM (``lstm_baseline``) and the TFTS ``rnn``. The
training pipelines never reload those weights, so a corrupt save, a config
drift, or a model/config mismatch on rebuild would silently go unnoticed.

This script closes that gap. For every model present in ``results.json`` it:

- rebuilds the model with the exact constructor used at training time,
- loads each fold's saved ``.weights.h5`` into a fresh instance,
- re-runs validation inference on the same held-out breaths and fold indices,
- recomputes the per-fold and the global inspiration-MAE,
- and reports whether each reloaded global metric reproduces the value recorded
  in ``results.json`` (within ``--tolerance``).

Inference is deterministic for these models (all dropout rates are 0), so a
reloaded checkpoint must reproduce the training-time metric to within floating
point round-off; any larger difference indicates a broken save or a stale
rebuild. The verifier also compares the reloaded out-of-fold predictions against
the ``{model}_oof.npy`` snapshot saved during training when one exists.

Two feature pipelines are supported. The ``reference`` and ``bert`` models use
the engineered, robust-scaled features and the multi-channel target stack from
``VentilatorDataset.prepare_reference``. The ``rnn`` and ``lstm_baseline`` models
consume the raw five inputs and single pressure channel from
``VentilatorDataset.prepare_data``. The pipeline is selected automatically from
the models recorded in ``results.json``.

Run from ``Time-series-prediction/`` (engineered-feature pipelines):
    uv run python -m benchmark.verify_saved_models --data ../data/ventilator/train.csv
    uv run python -m benchmark.verify_saved_models --data ../data/ventilator/train.csv \
        --output outputs/ventilator_bert_6e
Raw-feature pipelines:
    uv run python -m benchmark.verify_saved_models --data ../data/ventilator/train.csv \
        --output outputs/ventilator_aligned_10e
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from benchmark.datasets.ventilator import VentilatorDataset, inspiration_mae
from benchmark.ventilator_models import ModelParameters, VentilatorRunError, build_ventilator_model

METRIC = "inspiration_mae"
VALID_BATCH = 256
# Models that consume engineered reference features and the multi-channel target stack.
_REFERENCE_MODELS = frozenset({"reference", "bert"})
# Models that consume raw five-element inputs and a single pressure channel.
_RAW_MODELS = frozenset({"rnn", "lstm_baseline"})


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of reloading and re-running one model's checkpoints."""

    model: str
    recorded: float
    reloaded: float
    max_abs_diff: float | None
    fold_metrics: list[float]
    ok: bool


def compare_metrics(recorded: float, reloaded: float, tolerance: float) -> bool:
    """True when the reloaded metric reproduces the recorded value.

    Inferences are deterministic (dropout is 0 for every layer in these models),
    so the two numbers must agree to within floating point round-off.
    """
    return bool(
        np.isfinite(recorded)
        and np.isfinite(reloaded)
        and np.isfinite(tolerance)
        and tolerance >= 0
        and abs(reloaded - recorded) <= tolerance
    )


def _rnn_overrides(output: Path, model: str) -> ModelParameters:
    """Return the RNNConfig keyword arguments used to train ``model``.

    Prefers the per-fold saved config (authoritative for what was trained);
    falls back to the manifest ``model_overrides`` entry.
    """
    config_path = output / f"{model}_fold0.weights.config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        keys = (
            "rnn_hidden_size",
            "rnn_type",
            "bi_direction",
            "dense_hidden_size",
            "num_stacked_layers",
            "scheduled_sampling",
            "use_attention",
            "return_sequences",
            "input_projection",
        )
        return {key: saved[key] for key in keys if key in saved}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("model_overrides", {}).get(model, {})
    return {}


def _build_model(name: str, num_features: int, output: Path) -> tf.keras.Model:
    """Reconstruct the exact model the training script used for ``name``."""
    if name == "rnn":
        return build_ventilator_model(name, num_features, _rnn_overrides(output, name))
    manifest_path = output / "manifest.json"
    overrides = {}
    if manifest_path.exists():
        overrides = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_overrides", {}).get(name, {})
    return build_ventilator_model(name, num_features, overrides)


def _load_recorded_metric(output: Path, model: str) -> float | None:
    """Return the global inspiration-MAE recorded in results.json for ``model``."""
    results_path = output / "results.json"
    if not results_path.exists():
        return None
    records = json.loads(results_path.read_text(encoding="utf-8"))
    for record in records:
        if record.get("model") == model:
            metrics = record.get("metrics", {})
            if METRIC in metrics:
                return float(metrics[METRIC])
    return None


def _load_folds(dataset: VentilatorDataset, output: Path) -> list[tuple[int, np.ndarray]]:
    """Load the recorded fold splits, falling back to a deterministic recompute."""
    return dataset.load_folds(output / "folds.npz")


def _load_pipeline(output: Path, data: str) -> tuple[VentilatorDataset, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the loaded dataset alongside arrays so fold recovery uses its IDs."""
    results_path = output / "results.json"
    records = json.loads(results_path.read_text(encoding="utf-8"))
    models = {record.get("model") for record in records if record.get("model")}
    if not models:
        raise VentilatorRunError(f"{results_path} lists no models to verify")
    if models - (_REFERENCE_MODELS | _RAW_MODELS):
        raise VentilatorRunError(f"Unknown models in results: {sorted(models)}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "data_sha256" in saved:
            with Path(data).open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            if checksum != saved["data_sha256"]:
                raise VentilatorRunError("Input CSV checksum differs from the saved run")
    uses_engineered = bool(models & _REFERENCE_MODELS)
    uses_raw = bool(models & _RAW_MODELS)
    if uses_engineered and uses_raw:
        raise VentilatorRunError(f"Mixed engineered and raw-feature models in {output}: {sorted(models)}")

    dataset = VentilatorDataset()
    if uses_engineered:
        x, targets, mask = dataset.prepare_reference(path=data)
        return dataset, x, targets, mask
    x, y = dataset.prepare_data(path=data)
    return dataset, x, y, dataset.mask


def verify_model(
    name: str,
    output: Path,
    x: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    folds: list[tuple[int, np.ndarray]],
    tolerance: float,
    seed: int = 0,
) -> VerificationResult:
    """Reload every fold checkpoint for ``name`` and compare the global metric.

    Raises ``FileNotFoundError`` if a recorded fold is missing its weights file,
    so a partial save is reported instead of silently passing.
    """
    num_features = x.shape[-1]
    out_of_fold = np.full_like(targets, np.nan)
    fold_metrics: list[float] = []
    for fold, valid in folds:
        weights_path = output / f"{name}_fold{fold}.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {name} fold {fold}: {weights_path}")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = _build_model(name, num_features, output)
        # Force a build so load_weights has variables to assign into.
        _ = model(x[valid[:2]], training=False)
        model.load_weights(str(weights_path))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        validation = tf.data.Dataset.from_tensor_slices(x[valid]).batch(VALID_BATCH).with_options(options)
        prediction = model.predict(validation, verbose=0)
        if prediction.shape != targets[valid].shape or not np.isfinite(prediction).all():
            raise VentilatorRunError(f"Invalid reloaded prediction shape or values for {name} fold {fold}")
        out_of_fold[valid] = prediction
        fold_metrics.append(inspiration_mae(targets[valid][..., 0:1], prediction[..., 0:1], mask[valid]))

    reloaded = inspiration_mae(targets[..., 0:1], out_of_fold[..., 0:1], mask)
    recorded = _load_recorded_metric(output, name)

    saved_oof = output / f"{name}_oof.npy"
    max_abs_diff: float | None = None
    if saved_oof.exists():
        saved = np.load(saved_oof)
        if saved.shape == out_of_fold.shape and np.isfinite(saved).all() and np.isfinite(out_of_fold).all():
            max_abs_diff = float(np.max(np.abs(out_of_fold - saved)))
        else:
            max_abs_diff = float("inf")

    ok = recorded is not None and compare_metrics(recorded, reloaded, tolerance)
    if max_abs_diff is not None:
        ok = ok and np.isfinite(max_abs_diff) and max_abs_diff <= tolerance

    return VerificationResult(
        model=name,
        recorded=recorded if recorded is not None else float("nan"),
        reloaded=reloaded,
        max_abs_diff=max_abs_diff,
        fold_metrics=fold_metrics,
        ok=bool(ok),
    )


def _print_result(result: VerificationResult, tolerance: float) -> None:
    recorded = f"{result.recorded:.6f}" if np.isfinite(result.recorded) else "N/A"
    print(f"\n  model: {result.model}")
    print(f"    recorded global inspiration MAE : {recorded}")
    print(f"    reloaded global inspiration MAE : {result.reloaded:.6f}")
    if result.max_abs_diff is not None:
        print(f"    reloaded vs saved oof max |diff| : {result.max_abs_diff:.3e}")
    else:
        print("    reloaded vs saved oof max |diff| : (no oof.npy snapshot)")
    if result.fold_metrics:
        per_fold = "  ".join(f"fold {fold}: {score:.6f}" for fold, score in enumerate(result.fold_metrics))
        print(f"    per-fold reloaded MAE            : {per_fold}")
    print(f"    result: {'PASS' if result.ok else 'FAIL'} (tolerance {tolerance:g})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="path to ventilator train.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/ventilator_bert"))
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max |reloaded - recorded| to pass")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    results_path = args.output / "results.json"
    if not args.output.exists() or not results_path.exists():
        parser.error(f"{args.output} contains no results.json. Run the benchmark first.")

    dataset, x, targets, mask = _load_pipeline(args.output, str(args.data))
    folds = _load_folds(dataset, args.output)

    records = json.loads(results_path.read_text(encoding="utf-8"))
    models = sorted({record.get("model") for record in records if record.get("model")})

    engineered = any(model in _REFERENCE_MODELS for model in models)
    print(f"Verifying {len(models)} model(s) over {len(folds)} fold(s) in {args.output}")
    print(f"feature pipeline: {'engineered reference' if engineered else 'raw five-element'}")
    failed = False
    for name in models:
        try:
            result = verify_model(name, args.output, x, targets, mask, folds, args.tolerance)
        except (FileNotFoundError, ValueError) as error:
            print(f"\n  model: {name}, ERROR: {error}")
            failed = True
            continue
        _print_result(result, args.tolerance)
        if not result.ok:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
