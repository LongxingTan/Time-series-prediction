"""CPU comparison using the benchmark configuration and result formats.

Run from the repository root with:
uv run python -m benchmark.ventilator --data ../data/ventilator/train.csv
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
from benchmark.ventilator_models import ALIGNED_TFTS_MODELS, VentilatorRunError, build_ventilator_model, ventilator_loss


def run_comparison(
    path: Path,
    config: BenchmarkConfig,
    threads: int = 4,
    resume: bool = False,
    folds_to_run: list[int] | None = None,
) -> BenchmarkResults:
    """Train both models on identical saved five-fold data using only the CPU.

    All five folds are required for an out-of-fold score. The final epoch is
    evaluated, matching the notebook, without validation-driven checkpointing.
    """
    if config.batch_size < 1 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise VentilatorRunError("Batch size and learning rate must be positive")
    if not config.models or set(config.models) - ({"rnn", "lstm_baseline"} | set(ALIGNED_TFTS_MODELS)):
        raise VentilatorRunError("Select rnn, lstm_baseline, or a listed aligned TFTS architecture")
    if folds_to_run is not None and (not resume or any(fold not in range(5) for fold in folds_to_run)):
        raise VentilatorRunError("Explicit folds require resume and indices between 0 and 4")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = VentilatorDataset()
    x, y = dataset.prepare_data(path=str(path))
    packed = np.concatenate([y, dataset.mask], axis=-1)
    validation_folds = (
        dataset.load_folds(output / "folds.npz")
        if resume
        else [(fold, dataset.fold_indices(fold)[1]) for fold in range(5)]
    )
    folds = [(np.setdiff1d(np.arange(len(x)), valid), valid) for _, valid in validation_folds]
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {
        "data_sha256": checksum,
        "features": ["R", "C", "u_in", "u_in_cumsum", "u_out"],
        "split": "GroupKFold(n_splits=5, shuffle=False)",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "model_overrides": config.per_model_config,
        "device": "CPU",
        "selection": "final epoch",
        "score": "global inspiration MAE",
    }
    manifest_path = output / "manifest.json"
    if resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("data_sha256", "epochs", "batch_size", "learning_rate", "seed", "model_overrides"):
            if saved[key] != manifest[key]:
                raise VentilatorRunError(f"Resume setting differs from saved manifest: {key}")
        if folds_to_run is not None:
            for name in config.models:
                for fold in set(range(5)) - set(folds_to_run):
                    if not (output / f"{name}_fold{fold}.weights.h5").exists():
                        raise VentilatorRunError(f"Missing checkpoint for unselected fold: {name} fold {fold}")
    else:
        if manifest_path.exists():
            raise VentilatorRunError("Use --resume or a fresh output directory")
        with manifest_path.open("w", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2)
    np.savez_compressed(
        output / "folds.npz",
        breath_ids=dataset.breath_ids,
        **{f"valid_{fold}": valid for fold, (_, valid) in enumerate(folds)},
    )
    results_path = output / "results.json"
    records = json.loads(results_path.read_text(encoding="utf-8")) if resume and results_path.exists() else []
    records = [record for record in records if record["model"] not in config.models]
    BenchmarkResults(records).to_json(str(results_path))
    BenchmarkResults(records).to_csv(str(output / "results.csv"))
    for model_name in config.models:
        predictions = np.full_like(y, np.nan)
        for fold, (train, valid) in enumerate(folds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(config.seed)
            model = build_ventilator_model(model_name, x.shape[-1], config.get_model_config(model_name))
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            weights_path = output / f"{model_name}_fold{fold}.weights.h5"
            train_fold = not resume or (fold in folds_to_run if folds_to_run is not None else not weights_path.exists())
            if train_fold:
                train_data = (
                    tf.data.Dataset.from_tensor_slices((x[train], packed[train]))
                    .shuffle(len(train), seed=config.seed, reshuffle_each_iteration=True)
                    .batch(config.batch_size, drop_remainder=True)
                    .with_options(options)
                )
                steps = len(train) // config.batch_size
                if steps == 0:
                    raise VentilatorRunError("Training fold must contain at least one full batch")
                schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                    config.learning_rate, decay_steps=config.epochs * steps, end_learning_rate=0.0
                )
                model.compile(optimizer=tf.keras.optimizers.Adam(schedule, epsilon=1e-8), loss=ventilator_loss)
                print(f"{model_name}: training fold {fold + 1}/5", flush=True)
                history = model.fit(train_data, epochs=config.epochs, verbose=config.verbose, shuffle=False)
                model.save_weights(str(weights_path))
                with (output / f"{model_name}_fold{fold}_history.json").open("w", encoding="utf-8") as destination:
                    json.dump(
                        {name: [float(value) for value in values] for name, values in history.history.items()},
                        destination,
                    )
            else:
                model(x[valid[:1]], training=False)
                model.load_weights(str(weights_path))
            valid_data = tf.data.Dataset.from_tensor_slices(x[valid]).batch(256).with_options(options)
            predictions[valid] = model.predict(valid_data, verbose=0)
            score = inspiration_mae(y[valid], predictions[valid], dataset.mask[valid])
            print(f"{model_name}: fold {fold + 1} inspiration MAE = {score:.6f}", flush=True)
            np.save(output / f"{model_name}_oof.npy", predictions)
        score = inspiration_mae(y, predictions, dataset.mask)
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


def resume_comparison(path: Path, output: Path, threads: int = 4, folds: list[int] | None = None) -> BenchmarkResults:
    """Resume an RNN run using its saved protocol and the normal training path."""
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = BenchmarkConfig(
        models=["rnn"],
        datasets=["ventilator"],
        epochs=saved["epochs"],
        batch_size=saved["batch_size"],
        learning_rate=saved["learning_rate"],
        seed=saved["seed"],
        per_model_config=saved["model_overrides"],
        output_dir=str(output),
        verbose=2,
    )
    return run_comparison(path, config, threads=threads, resume=True, folds_to_run=folds)


def main() -> None:
    """Expose the reproducible CPU comparison as a module command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--models", nargs="+", choices=["lstm_baseline", *ALIGNED_TFTS_MODELS])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true", help="Reload saved folds and train missing checkpoints")
    args = parser.parse_args()
    config = (
        BenchmarkConfig.from_yaml(args.config)
        if args.config
        else BenchmarkConfig(
            models=["lstm_baseline", "rnn"],
            datasets=["ventilator"],
            epochs=200,
            batch_size=128,
            output_dir="outputs/ventilator",
            verbose=2,
        )
    )
    config = replace(
        config,
        models=args.models if args.models is not None else config.models,
        epochs=args.epochs if args.epochs is not None else config.epochs,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        output_dir=args.output if args.output is not None else config.output_dir,
    )
    run_comparison(args.data, config, threads=args.threads, resume=args.resume)


if __name__ == "__main__":
    main()
