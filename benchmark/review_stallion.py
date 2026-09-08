"""Compare TiDE checkouts on the existing Stallion validation protocol.

Requires a local Stallion parquet (date, agency, sku, volume). No downloads.
The final six observations are validation, also used for checkpoint selection;
these are validation metrics, not an independent test-set estimate.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    import tensorflow as tf

    from tfts.models.tide import Tide, TideConfig

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    frame = pd.read_parquet(args.data).sort_values("date")
    series = [group.volume.to_numpy(dtype="float32") for _, group in frame.groupby(["agency", "sku"], sort=True)]
    if any(len(values) != 60 for values in series):
        raise ValueError("Expected 60 monthly observations per series")
    values = np.stack(series)
    logs = np.log1p(values)
    val_x = logs[:, -30:-6, None]
    val_y = values[:, -6:]
    rows = []
    for seed in (11, 29, 47):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        model = Tide(6, TideConfig())
        model(val_x[:2], training=False)
        optimizer = tf.keras.optimizers.Adam(0.001)

        @tf.function
        def train(x, y):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(model(x, training=True) - y))
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        @tf.function
        def predict(x):
            return model(x, training=False)

        best, patience, best_prediction = float("inf"), 6, None
        history = []
        for epoch in range(30):
            for _ in range(100):
                # Draw series/start pairs in a fixed order, shared by both checkouts.
                indices = rng.integers(len(values), size=64)
                starts = rng.integers(25, size=64)
                windows = logs[indices[:, None], starts[:, None] + np.arange(30)]
                train(windows[:, :24, None], windows[:, 24:, None])
            prediction = np.expm1(predict(val_x).numpy()[..., 0])
            mae = float(np.mean(np.abs(prediction - val_y)))
            history.append(mae)
            if mae < best:
                best, best_prediction, patience = mae, prediction, 6
            else:
                patience -= 1
            print(f"seed={seed} epoch={epoch + 1} mae={mae:.4f}", flush=True)
            if patience == 0:
                break
        row = dict(
            seed=seed,
            mae=best,
            mse=float(np.mean((best_prediction - val_y) ** 2)),
            smape=float(
                np.mean(
                    200 * np.abs(best_prediction - val_y) / np.maximum(np.abs(best_prediction) + np.abs(val_y), 1e-8)
                )
            ),
            epochs=len(history),
            validation_mae=history,
            parameters=model.count_params(),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
