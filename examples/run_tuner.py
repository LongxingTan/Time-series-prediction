"""Demo to tune the model parameters by Autotune"""

import numpy as np

from tfts import AutoConfig, AutoModel, AutoModelForAnomaly, KerasTrainer, get_data


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
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 10, 50)

        # Create model config
        config = AutoConfig.for_model(self.use_model)
        config.rnn_hidden_size = hidden_units
        config.num_stacked_layers = num_layers

        model = AutoModel.from_config(config, predict_sequence_length=self.predict_sequence_length)
        trainer = KerasTrainer(model, optimizer_config={"learning_rate": learning_rate})

        trainer.train(self.train_data, self.valid_data, epochs=epochs, verbose=0)

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


if __name__ == "__main__":
    train_length = 24
    predict_sequence_length = 8
    (x_train, y_train), (x_valid, y_valid) = get_data("sine", train_length, predict_sequence_length, test_size=0.2)

    tuner = AutoTuner(
        use_model="rnn",
        train_data=(x_train, y_train),
        valid_data=(x_valid, y_valid),
        predict_sequence_length=predict_sequence_length,
    )

    study = tuner.run(n_trials=20, direction="minimize")
