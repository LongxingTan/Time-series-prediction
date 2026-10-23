"""Forecasting CLI and end-to-end forecasting workflow.

Provides the highest-level API for time series forecasting, wrapping
DataProcessor, AutoModel, and Trainer behind a single interface.
"""

import argparse
import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from ..data import get_data
from ..data.processor import DataProcessor
from ..models.auto_config import AutoConfig
from ..models.auto_model import AutoModel
from ..tasks.pipeline import TaskPipeline
from ..training import Trainer, set_seed

logger = logging.getLogger(__name__)

__all__ = ["ForecastingPipeline", "main"]


class ForecastingPipeline(TaskPipeline):
    """Deprecated forecasting argument adapter over the shared estimator."""

    def __init__(
        self,
        model="rnn",
        lookback=96,
        horizon=24,
        config=None,
        batch_size=32,
        normalize="standard",
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=5,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            "forecasting",
            model,
            config=config,
            batch_size=batch_size,
            normalize=normalize,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            **kwargs,
        )
        self._estimator.context, self._estimator.horizon = lookback, horizon

    @property
    def lookback(self):
        return getattr(self._estimator, "context_", self._estimator.context)

    @property
    def horizon(self):
        return getattr(self._estimator, "horizon_", self._estimator.horizon)

    def fit(
        self, df, targets=None, target_col=None, time_col=None, validation_split=0.2, epochs=None, verbose=1, **kwargs
    ):
        if epochs is not None:
            kwargs["epochs"] = epochs
        if not isinstance(df, pd.DataFrame):
            return super().fit(df, targets, verbose=verbose, **kwargs)
        from tfts.features import TimeSeriesSchema

        if targets is not None:
            if target_col is not None or not isinstance(targets, str):
                raise ValueError("Use target_col for dataframe training")
            target_col = targets
        df = df.copy()
        time_col = time_col or DataProcessor._infer_time(df)
        target_col = target_col or DataProcessor._infer_target(df)
        self._estimator.schema = TimeSeriesSchema(time_col, (target_col,))
        self._estimator.validation = validation_split
        self._estimator.fit(df, verbose=verbose, **kwargs)
        return self._estimator.history_

    def predict(self, inputs=None, steps=None, df=None):
        if df is not None:
            if inputs is not None:
                raise ValueError("Pass prediction history once")
            inputs = df
        if inputs is None:
            raise ValueError("Pass recent history as inputs or df")
        if getattr(self._estimator, "input_kind_", None) == "frame":
            output = self._estimator.predict_output(inputs)
            values = np.asarray(output.predictions)
            if steps is not None and (steps < 1 or steps > self.horizon):
                raise ValueError("steps must be between 1 and the fitted horizon")
            return values[:, :steps] if steps is not None else values
        return self._estimator.predict(inputs, steps=steps)

    def evaluate(self, df, target_col=None, time_col=None):
        return self._estimator.evaluate(df)

    def summary(self):
        self.model.summary()


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
