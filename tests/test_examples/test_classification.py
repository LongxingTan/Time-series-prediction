from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf


def _synthetic_classification_data(n=256, length=32, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0.0, 1.0, (n // 2, length, 1)), rng.normal(2.0, 1.0, (n // 2, length, 1))])
    y = np.array([0] * (n // 2) + [1] * (n // 2)).astype(int)
    return x.astype(np.float32), y


class ClassificationExampleTest(unittest.TestCase):
    def test_run_train_with_synthetic_data(self):
        """Exercise the example's run_train code path without downloading FordA."""
        from examples import run_classification as example
        from tfts import set_seed

        x, y = _synthetic_classification_data()
        x_test, y_test = _synthetic_classification_data(seed=1)
        args = type(
            "args",
            (),
            {
                "seed": 315,
                "use_model": "bert",
                "num_labels": 2,
                "epochs": 2,
                "batch_size": 64,
                "learning_rate": 2e-4,
            },
        )()

        with patch.object(example, "prepare_data", return_value=(x, y, x_test, y_test)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                set_seed(args.seed)
                example.run_train(args)
        self.assertIn("[[", buf.getvalue())  # confusion matrix was printed

    def test_classifier_actually_learns(self):
        """The classification example uses logits; verify the model can learn."""
        from tfts import AutoConfig, AutoModelForClassification, KerasTrainer, set_seed

        x, y = _synthetic_classification_data(n=512)
        from sklearn.model_selection import train_test_split

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        set_seed(315)
        config = AutoConfig.for_model("bert")
        model = AutoModelForClassification.from_config(config, num_labels=2)
        trainer = KerasTrainer(model)
        trainer.train(
            (x_train, y_train),
            valid_dataset=(x_val, y_val),
            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(2e-4),
            epochs=3,
            batch_size=64,
            metrics=["sparse_categorical_accuracy"],
            verbose=0,
        )
        y_pred = model(x_val)
        acc = float(np.mean(np.argmax(y_pred, axis=1) == y_val))
        self.assertGreater(acc, 0.6)


if __name__ == "__main__":
    unittest.main()
