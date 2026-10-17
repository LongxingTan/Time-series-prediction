"""CPU comparison of cdeotte's reference transformer against the TFTS BERT.

The reference is cdeotte's "TensorFlow Transformer [0.112]" notebook. Its
protocol is reproduced on the engineered features the notebook uses:

- ``VentilatorDataset.prepare_reference`` builds the notebook's 64-wide feature
  frame (``add_features`` plus one-hot R/C/R__C), applies a manual RobustScaler,
  and returns the three-channel target stack ``[pressure, pressure_diff,
  pressure_integral]`` together with the inspiration mask.
- Folds are the same unshuffled five-fold breath split used everywhere else in
  the benchmark, so every model sees identical held-out breaths.
- The training loss is ``masked_multi_mae`` (masked MAE weighted to inspiration
  rows, matching the notebook's transient sample weighting).
- The score is the global inspiration MAE on the pressure channel only, which is
  the channel actually submitted to Kaggle.

The reference model is ``ReferenceTransformer`` (12 blocks, embed=64, heads=8,
ff=128, dropout=0, 0.7/0.3 skip connections). The comparator is TFTS ``Bert``
with its encoder-only per-timestep ``regression_head``, which is the TFTS model
whose architecture matches the reference notebook (an encoder without a
decoder). Both models are trained on CPU with the same data, folds, loss,
optimizer schedule, epochs and batch size, and the final epoch is evaluated.

Run from ``Time-series-prediction/``:
    uv run python -m benchmark.ventilator_bert --data ../data/ventilator/train.csv
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from benchmark.base import BenchmarkConfig
from benchmark.datasets.ventilator import VentilatorDataset, inspiration_mae
from benchmark.formatter import BenchmarkResults
from benchmark.ventilator_models import VentilatorRunError, build_ventilator_model, masked_multi_mae


def _segments(epochs: int) -> list[int]:
    """Three increasing cosine-with-restart segments summing to ``epochs``."""
    if epochs < 1:
        raise VentilatorRunError("epochs must be positive")
    if epochs < 3:
        return [1] * epochs
    n1 = max(1, epochs // 6)
    n3 = max(1, epochs // 2)
    n2 = epochs - n1 - n3
    if n2 < 1:
        n2 = 1
        n1 = max(1, (epochs - n2) // 2)
        n3 = epochs - n1 - n2
    return [n1, n2, n3]


def _lr_trace(epochs: int, start: float = 1e-6, peak: float = 6e-4) -> np.ndarray:
    """The notebook's cosine-with-restarts learning-rate trace over ``epochs``."""
    segments = _segments(epochs)
    boundaries = np.cumsum([0] + segments)

    def lrfn(epoch: int) -> float:
        for idx, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if lo <= epoch < hi or (hi == boundaries[-1] and epoch >= lo):
                width = segments[idx]
                if idx == len(segments) - 1 and epoch >= hi:
                    epoch_idx = width - 1
                else:
                    epoch_idx = epoch - lo
                phase = np.pi * epoch_idx / max(width - 1, 1)
                return (peak - start) * 0.5 * (1 + np.cos(phase)) + start
        return start

    return np.asarray([lrfn(epoch) for epoch in range(epochs)], dtype=np.float64)


def _lr_schedule(epochs: int, start: float = 1e-6, peak: float = 6e-4) -> tf.keras.callbacks.LearningRateScheduler:
    """A LearningRateScheduler callback matching the precomputed trace.

    The reference notebook uses three cosine restarts. The same three-segment
    structure is spread across the configured CPU epoch budget so the reference
    and the TFTS BERT share an identical schedule.
    """
    trace = _lr_trace(epochs, start, peak)
    return tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(trace[int(epoch)]), verbose=0)


def run_comparison(path: Path, config: BenchmarkConfig, threads: int = 6, folds_to_run: int = 5) -> BenchmarkResults:
    """Train the reference transformer and the TFTS BERT on identical folds.

    ``folds_to_run`` caps how many of the five folds are executed and is used
    only for quick pipeline checks; a full score needs all five.
    """
    if not 1 <= folds_to_run <= 5:
        raise VentilatorRunError("folds must be between 1 and 5")
    if config.batch_size < 1 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise VentilatorRunError("Batch size and learning rate must be positive")
    if not config.models or set(config.models) - {"reference", "bert"}:
        raise VentilatorRunError("Select reference and/or bert")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    output = Path(config.output_dir)
    if (output / "manifest.json").exists():
        raise VentilatorRunError("Use a fresh output directory for a new transformer run")
    output.mkdir(parents=True, exist_ok=True)

    dataset = VentilatorDataset()
    x, targets, mask = dataset.prepare_reference(path=str(path))
    num_features = x.shape[-1]
    for name in config.models:
        build_ventilator_model(name, num_features, config.get_model_config(name))
    packed = np.concatenate([targets, mask], axis=-1)
    folds = [dataset.fold_indices(fold) for fold in range(folds_to_run)]
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    lr_trace = _lr_trace(config.epochs, peak=config.learning_rate)
    manifest = {
        "data_sha256": checksum,
        "features": ["engineered (add_features + one-hot R/C/R__C), RobustScaled"],
        "num_features": num_features,
        "preprocessing": "robust-unit-zero-iqr-v1",
        "targets": ["pressure", "pressure_diff", "pressure_integral"],
        "split": "GroupKFold(n_splits=5, shuffle=False)",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": "cosine-with-restarts (3 segments)",
        "lr_peak": config.learning_rate,
        "lr_trace_first_peak_max_last": [float(lr_trace[0]), float(lr_trace.max()), float(lr_trace[-1])],
        "seed": config.seed,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "device": "CPU",
        "selection": "final epoch",
        "score": "global inspiration MAE (pressure channel)",
        "model_overrides": config.per_model_config,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
    np.savez_compressed(
        output / "folds.npz",
        breath_ids=dataset.breath_ids,
        **{f"valid_{fold}": valid for fold, (_, valid) in enumerate(folds)},
    )

    records = []
    for model_name in config.models:
        predictions = np.full_like(targets, np.nan)
        for fold, (train, valid) in enumerate(folds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(config.seed)
            model = build_ventilator_model(model_name, num_features, config.get_model_config(model_name))
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            train_data = (
                tf.data.Dataset.from_tensor_slices((x[train], packed[train]))
                .shuffle(len(train), seed=config.seed, reshuffle_each_iteration=True)
                .batch(config.batch_size, drop_remainder=True)
                .with_options(options)
            )
            steps = len(train) // config.batch_size
            if steps == 0:
                raise VentilatorRunError("Training fold must contain at least one full batch")
            model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate, epsilon=1e-8), loss=masked_multi_mae)
            print(f"{model_name}: fold {fold + 1}/5", flush=True)
            history = model.fit(
                train_data,
                epochs=config.epochs,
                verbose=config.verbose,
                shuffle=False,
                callbacks=[_lr_schedule(config.epochs, peak=config.learning_rate)],
            )
            valid_data = tf.data.Dataset.from_tensor_slices(x[valid]).batch(256).with_options(options)
            pred = model.predict(valid_data, verbose=0)
            predictions[valid] = pred
            score = inspiration_mae(targets[valid][..., 0:1], pred[..., 0:1], mask[valid])
            model.save_weights(str(output / f"{model_name}_fold{fold}.weights.h5"))
            with (output / f"{model_name}_fold{fold}_history.json").open("w", encoding="utf-8") as destination:
                json.dump(history.history, destination)
            print(f"{model_name}: fold {fold + 1} inspiration MAE = {score:.6f}", flush=True)
            np.save(output / f"{model_name}_oof.npy", predictions)
        if np.isfinite(predictions).all():
            score = inspiration_mae(targets[..., 0:1], predictions[..., 0:1], mask)
            records.append(
                {
                    "dataset": dataset.name,
                    "model": model_name,
                    "run": 0,
                    "seed": config.seed,
                    "metrics": {"inspiration_mae": score},
                }
            )
        results = BenchmarkResults(records)
        results.to_json(str(output / "results.json"))
        results.to_csv(str(output / "results.csv"))
    return BenchmarkResults(records)


def main() -> None:
    """Expose the reproducible CPU comparison as a module command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--models", nargs="+", choices=["reference", "bert"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.config:
        config = BenchmarkConfig.from_yaml(args.config)
    else:
        config = BenchmarkConfig(
            models=["reference", "bert"],
            datasets=["ventilator"],
            epochs=6,
            batch_size=64,
            learning_rate=6e-4,
            output_dir="outputs/ventilator_bert",
            verbose=1,
        )
    config = replace(
        config,
        models=args.models if args.models is not None else config.models,
        epochs=args.epochs if args.epochs is not None else config.epochs,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        output_dir=args.output if args.output is not None else config.output_dir,
    )
    run_comparison(args.data, config, threads=args.threads, folds_to_run=args.folds)


if __name__ == "__main__":
    main()
