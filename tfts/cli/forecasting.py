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
from ..trainer import Trainer, set_seed

logger = logging.getLogger(__name__)

__all__ = ["ForecastingPipeline", "main"]


class ForecastingPipeline:
    """End-to-end forecasting pipeline with a transformers-like API.

    Handles data preparation, model building, training, prediction, and
    evaluation in a single object.

    Args:
        model: Model name (e.g. ``'patch_tst'``, ``'transformer'``) or an
            ``AutoModel`` instance.
        lookback: Number of past time steps used as input.
        horizon: Number of future time steps to predict.
        config: Optional AutoConfig for fine-grained control.
        batch_size: Batch size for training.
        normalize: Normalization — ``'minmax'``, ``'standard'``, or ``None``.
        learning_rate: Learning rate for the optimizer.
        epochs: Default number of training epochs.
        early_stopping_patience: Patience for early stopping.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments passed to the model config.

    Examples:
        >>> import tfts
        >>> pipe = tfts.pipeline("forecasting", model="dlinear",
        ...                      lookback=96, horizon=24)
        >>> pipe.fit(df, target_col="value", epochs=50)
        >>> preds = pipe.predict(steps=24)

        >>> # Evaluate on holdout
        >>> metrics = pipe.evaluate(test_df)
    """

    def __init__(
        self,
        model: Union[str, AutoModel] = "dlinear",
        lookback: int = 96,
        horizon: int = 24,
        config: Optional[AutoConfig] = None,
        batch_size: int = 32,
        normalize: Optional[str] = "minmax",
        learning_rate: float = 1e-3,
        epochs: int = 50,
        early_stopping_patience: Optional[int] = 5,
        seed: int = 42,
        **kwargs,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        set_seed(seed)

        # Model setup
        if isinstance(model, str):
            self.config = config or AutoConfig.for_model(model)
            self.config.update(kwargs)
            self._model = AutoModel.from_config(self.config, predict_sequence_length=horizon)
        else:
            self._model = model
            self.config = getattr(model, "config", config)

        self.model_name = getattr(self.config, "model_type", model if isinstance(model, str) else "custom")

        # Data processor
        self.processor = DataProcessor(
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
            normalize=normalize,
        )

        # Will be set during fit()
        self.trainer: Trainer = Trainer(self._model)
        self._target_col: Optional[str] = None
        self._time_col: Optional[str] = None
        self._fitted: bool = False

        logger.info(f"Pipeline ready: model={self.model_name}, lookback={lookback}, horizon={horizon}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        validation_split: float = 0.2,
        epochs: Optional[int] = None,
        verbose: int = 1,
        **trainer_kwargs,
    ) -> tf.keras.callbacks.History:
        """Train the forecasting model.

        Args:
            df: Time series DataFrame.
            target_col: Column to forecast. Auto-detected if None.
            time_col: Time column. Auto-detected if None.
            validation_split: Fraction of data for validation.
            epochs: Training epochs (overrides constructor default).
            verbose: 0=silent, 1=progress bar, 2=one line.
            **trainer_kwargs: Passed to ``Trainer.train()``.

        Returns:
            Keras History object.
        """
        self._target_col = target_col
        self._time_col = time_col

        # Prepare data
        self.processor.validation_split = validation_split
        result = self.processor.prepare(df, target_col=target_col, time_col=time_col)

        if isinstance(result, tuple):
            train_ds, valid_ds = result
        else:
            train_ds, valid_ds = result, None

        # Train
        epochs = epochs or self.epochs
        history = self.trainer.train(
            train_ds,
            valid_dataset=valid_ds,
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            early_stopping_patience=self.early_stopping_patience,
            **{k: v for k, v in trainer_kwargs.items() if k != "early_stopping_patience"},
        )

        self._fitted = True
        return history

    def predict(self, steps: Optional[int] = None, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate forecasts.

        Args:
            steps: Number of steps to predict. Defaults to ``self.horizon``.
            df: DataFrame with new data to predict from (uses training data if None).

        Returns:
            Numpy array of shape ``(n_series, steps, n_targets)``.
        """
        if not self._fitted and self.trainer is None:
            raise RuntimeError("Pipeline must be fitted before prediction. Call .fit() first.")

        steps = steps or self.horizon

        if df is not None:
            ds = self.processor.prepare_for_inference(df, target_col=self._target_col, time_col=self._time_col)
            preds = self.trainer.predict(ds)
        else:
            # Use the model directly — user is responsible for input shape
            raise ValueError(
                "Please pass `df` with the latest lookback-length data for prediction, e.g. "
                "pipeline.predict(steps=24, df=recent_data)"
            )

        # Inverse transform
        preds = self.processor.inverse_transform(preds)

        # Trim to requested steps
        if preds.shape[1] > steps:
            preds = preds[:, :steps, :]

        return preds

    def evaluate(
        self, df: pd.DataFrame, target_col: Optional[str] = None, time_col: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate the model on a test DataFrame.

        Args:
            df: Test DataFrame.
            target_col: Target column name.
            time_col: Time column name.

        Returns:
            Dict of metric_name -> value.
        """
        if self.trainer is None:
            raise RuntimeError("Pipeline must be fitted before evaluation. Call .fit() first.")

        # Prepare the test data using the same processor params
        processor = DataProcessor(
            lookback=self.lookback, horizon=self.horizon, batch_size=self.batch_size, normalize=None
        )
        ds = processor.prepare(df, target_col=target_col or self._target_col, time_col=time_col or self._time_col)
        if isinstance(ds, tuple):
            ds = ds[0]  # use first split

        return self.trainer.evaluate(ds)

    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.trainer is None:
            raise RuntimeError("Nothing to save — train the pipeline first.")
        self.trainer.save_model(path)
        logger.info(f"Pipeline saved to {path}")

    def summary(self) -> None:
        """Print a summary of the pipeline."""
        line = "=" * 60
        print(line)
        print("TFTS Forecasting Pipeline")
        print(line)
        print(f"  Model      : {self.model_name}")
        print(f"  Lookback   : {self.lookback}")
        print(f"  Horizon    : {self.horizon}")
        print(f"  Batch size : {self.batch_size}")
        print(f"  Normalize  : {self.normalize}")
        print(f"  Fitted     : {self._fitted}")
        print(line)


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
    model = AutoModel.from_config(config, predict_sequence_length=args.horizon)
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
