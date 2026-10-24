"""Train and evaluate a forecasting model from the command line."""

import argparse
import logging
from typing import Optional, Sequence

import tensorflow as tf

from ..data import get_data
from ..models.auto_config import AutoConfig
from ..models.auto_model import AutoModel
from ..training import Trainer, set_seed

logger = logging.getLogger(__name__)

__all__ = ["main"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse forecasting CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a TFTS forecasting model on a built-in dataset.")
    parser.add_argument("--model", type=str, default="dlinear", help="Model name, e.g. dlinear, rnn, transformer.")
    parser.add_argument("--data", type=str, default="sine", help="Built-in dataset name, e.g. sine or airpassengers.")
    parser.add_argument("--lookback", type=int, default=24, help="Input sequence length.")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction sequence length.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=315, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for saving model weights.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Keras training verbosity.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Train and evaluate a forecasting model from the command line."""
    args = parse_args(argv)
    set_seed(args.seed)

    train_data, valid_data = get_data(
        args.data,
        train_length=args.lookback,
        predict_sequence_length=args.horizon,
        test_size=args.test_size,
    )

    config = AutoConfig.for_model(args.model)
    model = AutoModel.from_config(config, output_chunk_length=args.horizon)
    trainer = Trainer(model)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    trainer.train(
        train_data,
        valid_data,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        early_stopping_patience=args.early_stopping_patience,
    )

    metrics = trainer.evaluate(valid_data)
    preds = trainer.predict(valid_data[0])
    print(f"metrics: {metrics}")
    print(f"predictions shape: {preds.shape}")

    if args.output_dir:
        trainer.save_model(args.output_dir)
        print(f"saved model to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
