"""End-to-end time-series hyperparameter tuning with Optuna."""

import argparse
import os
import tempfile
from typing import Optional

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModel, KerasTrainer, get_data, set_seed


def parse_args():
    """Parse command-line options for the tuning example."""
    parser = argparse.ArgumentParser(description="Tune a TFTS forecasting model and restore the best trial.")
    parser.add_argument("--use_model", choices=("rnn",), default="rnn", help="backbone model to tune")
    parser.add_argument("--n_trials", type=int, default=20, help="number of Optuna trials")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/tuner",
        help="directory in which to save trial model artifacts",
    )
    parser.add_argument("--seed", type=int, default=315, help="random seed")
    return parser.parse_args()


class AutoTuner(object):
    """Tune model and training parameters with Optuna.

    Each trial is saved as a task-aware TFTS artifact. This makes the best
    trial immediately reusable for inference or further fine-tuning.
    """

    def __init__(
        self,
        use_model: str,
        train_data,
        valid_data=None,
        predict_sequence_length: int = 1,
        output_dir: Optional[str] = None,
    ) -> None:
        if use_model != "rnn":
            raise ValueError("AutoTuner currently defines an RNN-specific search space")
        self.use_model = use_model
        self.train_data = train_data
        self.valid_data = valid_data
        self.predict_sequence_length = predict_sequence_length
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="tfts_tuner_")

    def _trial_output_dir(self, trial_number: int) -> str:
        """Return the directory used for one trial's saved model."""
        return os.path.join(self.output_dir, "trials", f"trial_{trial_number}")

    def objective(self, trial):
        """Objective function to minimize or maximize."""
        # Suggest model configuration parameters
        hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 4)

        # Suggest training parameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 10, 50)

        # Create model config
        config = AutoConfig.for_model(self.use_model)
        config.rnn_hidden_size = hidden_units
        config.num_stacked_layers = num_layers

        model = AutoModel.from_config(config, predict_sequence_length=self.predict_sequence_length)
        trainer = KerasTrainer(model)

        trainer.train(
            self.train_data,
            self.valid_data,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            epochs=epochs,
            verbose=0,
        )

        trial_number = getattr(trial, "number", 0)
        trainer.save_model(self._trial_output_dir(trial_number))

        x_valid, y_valid = self.valid_data
        predictions = trainer.predict(x_valid)
        mse = np.mean((y_valid - predictions) ** 2)
        return mse

    def run(self, n_trials: int = 50, direction: str = "minimize"):
        """Run the tuning process."""
        import optuna

        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        return study

    def load_best_model(self, study):
        """Restore the best trial artifact for a final inference pass."""
        trial_dir = self._trial_output_dir(study.best_trial.number)
        x_train, _ = self.train_data
        return AutoModel.from_pretrained(trial_dir, sample_batch=x_train[:1])


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train_length = 24
    predict_sequence_length = 8
    (x_train, y_train), (x_valid, y_valid) = get_data("sine", train_length, predict_sequence_length, test_size=0.2)

    tuner = AutoTuner(
        use_model=args.use_model,
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        predict_sequence_length=predict_sequence_length,
        output_dir=args.output_dir,
    )

    study = tuner.run(n_trials=args.n_trials, direction="minimize")
    restored_model = tuner.load_best_model(study)
    predictions = restored_model(x_valid, training=False).numpy()
    print(f"Loaded best trial model, inference shape: {predictions.shape}")
