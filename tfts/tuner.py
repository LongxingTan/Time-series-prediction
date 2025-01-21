"""tfts auto tuner"""

import numpy as np

from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.trainer import KerasTrainer

__all__ = ["AutoTuner"]


class AutoTuner(object):
    """Auto tune parameters by optuna"""

    def __init__(self, use_model: str, train_data, valid_data=None, predict_sequence_length: int = 1) -> None:
        self.use_model = use_model
        self.train_data = train_data
        self.valid_data = valid_data
        self.predict_sequence_length = predict_sequence_length

    def objective(self, trial):
        """Objective function to minimize or maximize."""
        # Suggest model configuration parameters
        hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 4)

        # Suggest training parameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 5, 50)

        # Create model config
        config = AutoConfig.for_model(
            self.use_model,
            hidden_units=hidden_units,
            num_layers=num_layers,
        )

        # Create model and trainer
        model = AutoModel.from_config(config, predict_sequence_length=self.predict_sequence_length)
        trainer = KerasTrainer(model, optimizer_config={"learning_rate": learning_rate})

        # Train the model
        trainer.train(self.train_data, self.valid_data, epochs=epochs, verbose=0)

        # Evaluate the model (e.g., mean squared error)
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
