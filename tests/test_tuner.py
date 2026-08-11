import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from tfts.tuner import optuna_tuner
from tfts.tuner.optuna_tuner import OptunaTuner


class FakeTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        return (low, high, log)

    def suggest_int(self, name, low, high):
        return low


class OptunaTunerTest(unittest.TestCase):
    def setUp(self):
        data = (np.zeros((2, 4, 1)), np.zeros((2, 1, 1)))
        self.tuner = OptunaTuner(data, data, predict_sequence_length=1)

    def test_parameter_suggestions_cover_supported_specs(self):
        params = self.tuner._suggest_params(
            FakeTrial(),
            {
                "model_type": ["rnn", "dlinear"],
                "learning_rate": [0.0001, 0.01],
                "dropout": [1.0, 2.0],
                "num_layers": [1, 3],
            },
        )
        self.assertEqual(params["model_type"], "rnn")
        self.assertEqual(params["learning_rate"], (0.0001, 0.01, True))
        self.assertEqual(params["dropout"], (1.0, 2.0, False))
        self.assertEqual(params["num_layers"], 1)

        with self.assertRaises(ValueError):
            self.tuner._suggest_params(FakeTrial(), {"bad": [1, 2, 3]})

    def test_best_accessors_and_score_extraction(self):
        self.assertIsNone(self.tuner.get_study())
        with self.assertRaisesRegex(RuntimeError, "No search"):
            self.tuner.get_best_params()
        with self.assertRaisesRegex(RuntimeError, "No search"):
            self.tuner.get_best_score()

        self.assertEqual(self.tuner._extract_score({"mse": 0.25}), 0.25)
        callable_tuner = OptunaTuner(self.tuner.train_data, self.tuner.valid_data, metric=lambda y, p: 1.0)
        self.assertEqual(callable_tuner._extract_score({"mae": 0.5}), 0.5)
        with self.assertRaises(ValueError):
            self.tuner._extract_score({})
        self.assertEqual(repr(self.tuner), "OptunaTuner(metric='mse', direction='minimize')")

    def test_objective_builds_and_evaluates_a_trial(self):
        config = type("Config", (), {"hidden_size": 8})()
        fake_trainer = Mock()
        fake_trainer.evaluate.return_value = {"mse": 0.125}
        with patch.object(optuna_tuner.AutoConfig, "for_model", return_value=config), patch.object(
            optuna_tuner.AutoModel, "from_config", return_value=object()
        ), patch.object(optuna_tuner, "Trainer", return_value=fake_trainer), patch.object(
            optuna_tuner, "_default_optimizer", return_value=Mock()
        ):
            score = self.tuner._objective(
                FakeTrial(), {"model_type": ["rnn"], "hidden_size": [4, 8], "unknown": [1, 2]}, 1, 0
            )

        self.assertEqual(score, 0.125)
        self.assertEqual(config.hidden_size, 4)
        fake_trainer.train.assert_called_once()

    def test_search_stores_study_and_supports_optional_dependency_error(self):
        study = Mock()
        study.best_params = {"model_type": "rnn"}
        study.best_value = 0.2
        fake_optuna = Mock()
        fake_optuna.logging.WARNING = "warning"
        fake_optuna.create_study.return_value = study
        with patch.object(optuna_tuner, "_require_optuna", return_value=fake_optuna), patch.object(
            self.tuner, "_objective", return_value=0.2
        ):
            self.assertEqual(self.tuner.search({"model_type": ["rnn"]}, n_trials=1), ({"model_type": "rnn"}, 0.2))
        self.assertIs(self.tuner.get_study(), study)
        self.assertEqual(self.tuner.get_best_params(), {"model_type": "rnn"})
        self.assertEqual(self.tuner.get_best_score(), 0.2)

        with self.assertRaisesRegex(ImportError, "optuna is required"):
            optuna_tuner._require_optuna()

    def test_default_optimizer_is_a_tensorflow_optimizer(self):
        optimizer = optuna_tuner._default_optimizer(0.001)
        self.assertIsInstance(optimizer, tf.keras.optimizers.Optimizer)


if __name__ == "__main__":
    unittest.main()
