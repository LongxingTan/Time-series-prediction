"""Short single-fold pilot to compare RNN capacity choices.

Trains fold 0 only for a short budget and reports final train loss plus the
fold-0 inspiration MAE, so a capacity direction can be validated before the
full five-fold run. This is a development probe, not a benchmark result.
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from benchmark.datasets.ventilator import VentilatorDataset, inspiration_mae
from benchmark.ventilator_models import ventilator_loss
from tfts.models.rnn import RNN, RNNConfig


def pilot(path, configs, epochs, batch_size, seed, threads, output):
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    dataset = VentilatorDataset()
    x, y = dataset.prepare_data(path=str(path))
    packed = np.concatenate([y, dataset.mask], axis=-1)
    train, valid = dataset.fold_indices(0)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    out = {}
    for name, cfg in configs.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = RNN(80, RNNConfig(**cfg))
        model.build((None, 80, 5))
        _ = model(tf.zeros((1, 80, 5)), training=False)
        train_data = (
            tf.data.Dataset.from_tensor_slices((x[train], packed[train]))
            .shuffle(len(train), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .with_options(options)
        )
        steps = len(train) // batch_size
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            0.001, decay_steps=epochs * steps, end_learning_rate=0.0
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(schedule, epsilon=1e-8), loss=ventilator_loss)
        print(f"pilot {name}: {model.count_params()} params, fold0, {epochs} epochs", flush=True)
        history = model.fit(train_data, epochs=epochs, verbose=2, shuffle=False)
        valid_data = tf.data.Dataset.from_tensor_slices(x[valid]).batch(256).with_options(options)
        preds = model.predict(valid_data, verbose=0)
        score = inspiration_mae(y[valid], preds, dataset.mask[valid])
        print(f"pilot {name}: final train loss={history.history['loss'][-1]:.4f} fold0 MAE={score:.4f}", flush=True)
        out[name] = {
            "params": model.count_params(),
            "final_train_loss": history.history["loss"][-1],
            "full_train_loss": history.history["loss"],
            "fold0_mae": score,
            "config": cfg,
        }
    with open(output, "w", encoding="utf-8") as destination:
        json.dump(out, destination, indent=2)
    print(f"pilot written to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=str, default="/tmp/ventilator_pilot.json")
    args = parser.parse_args()
    configs = {
        "big_11M": {
            "rnn_hidden_size": 512,
            "rnn_type": "lstm",
            "bi_direction": True,
            "dense_hidden_size": 512,
            "num_stacked_layers": 2,
            "return_sequences": True,
            "input_projection": True,
        },
        "match_5M": {
            "rnn_hidden_size": 512,
            "rnn_type": "lstm",
            "bi_direction": True,
            "dense_hidden_size": 512,
            "num_stacked_layers": 1,
            "return_sequences": True,
            "input_projection": True,
        },
    }
    pilot(args.data, configs, args.epochs, args.batch_size, args.seed, args.threads, args.output)


if __name__ == "__main__":
    main()
