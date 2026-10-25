import unittest
from unittest.mock import patch

import tensorflow as tf

from examples.run_prediction_simple import parse_args, run_estimator, run_manual
from tfts import set_seed


def _make_args(**overrides):
    defaults = dict(
        seed=315,
        use_data="sine",
        use_model="dlinear",
        train_length=10,
        predict_sequence_length=5,
        epochs=1,
        batch_size=16,
        learning_rate=0.003,
        strategy="default",
    )
    defaults.update(overrides)
    return type("args", (), defaults)()


class PredictionTest(unittest.TestCase):
    def test_parse_args(self):
        with patch("sys.argv", ["parse_args", "--seed", "315"]):
            args = parse_args()
            self.assertEqual(args.seed, 315)

    def test_strategy_requires_manual(self):
        with patch("sys.argv", ["example", "--strategy", "mirrored"]):
            with self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)
        with patch("sys.argv", ["example", "--manual", "--strategy", "mirrored"]):
            self.assertEqual(parse_args().strategy, "mirrored")

    def test_train_manual(self):
        set_seed(315)
        run_manual(_make_args(use_model="rnn"))

    def test_estimator(self):
        set_seed(315)
        pred = run_estimator(_make_args(use_model="dlinear"))
        self.assertIsNotNone(pred)
        self.assertEqual(pred.shape[-1], 1)


if __name__ == "__main__":
    unittest.main()
