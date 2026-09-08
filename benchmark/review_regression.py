"""Reproducible synthetic accuracy/latency comparison for the model review.

Run each checkout in a separate process with the same arguments. This small
experiment is a regression diagnostic, not evidence of real-data accuracy.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    rows = []
    for name, config_name, class_name in [
        ("autoformer", "AutoFormerConfig", "AutoFormer"),
        ("tide", "TideConfig", "Tide"),
        ("itransformer", "ITransformerConfig", "ITransformer"),
        ("dlinear", "DLinearConfig", "DLinear"),
    ]:
        module = importlib.import_module("tfts.models." + name)
        for seed in (11, 29, 47):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            rng = np.random.default_rng(seed)
            phase = rng.uniform(-np.pi, np.pi, (160, 1, 1))
            amplitude = rng.uniform(0.5, 2, (160, 1, 1))
            offset = rng.normal(0, 1, (160, 1, 1))
            times = np.arange(32)[None, :, None]
            values = (offset + amplitude * np.sin(times * 2 * np.pi / 16 + phase)).astype("float32")
            x, y = values[:, :24], values[:, 24:]
            config = getattr(module, config_name)()
            config.update(
                dict(
                    hidden_size=16,
                    num_layers=1,
                    num_attention_heads=2,
                    ffn_intermediate_size=32,
                    hidden_dropout_prob=0.0,
                )
            )
            model = getattr(module, class_name)(predict_sequence_length=8, config=config)
            model(x[:16], training=False)
            optimizer = tf.keras.optimizers.Adam(0.001)

            @tf.function
            def train(a, b):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(model(a, training=True) - b))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            for step in range(args.steps):
                index = (step % 8) * 16
                train(x[index : index + 16], y[index : index + 16])

            @tf.function
            def predict(a):
                return model(a, training=False)

            predicted = predict(x[128:]).numpy()
            start = time.perf_counter()
            for _ in range(20):
                predict(x[128:]).numpy()
            row = dict(
                model=name,
                seed=seed,
                mse=float(np.mean((predicted - y[128:]) ** 2)),
                mae=float(np.mean(np.abs(predicted - y[128:]))),
                inference_ms=(time.perf_counter() - start) * 50,
                parameters=model.count_params(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
