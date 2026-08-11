"""TFTS Pipeline API — the primary entry point for users.

Provides a ``pipeline()`` factory function that returns the right
Pipeline subclass for the requested task.

Examples:
    >>> import tfts
    >>> forecaster = tfts.pipeline("forecasting", model="patch_tst",
    ...                            lookback=96, horizon=24)
    >>> forecaster.fit(df, target_col="sales")
    >>> preds = forecaster.predict(steps=24)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from ..models.auto_config import AutoConfig
from ..models.auto_model import AutoModel

if TYPE_CHECKING:
    from .forecasting import ForecastingPipeline

__all__ = ["pipeline", "ForecastingPipeline"]


def __getattr__(name: str):
    if name == "ForecastingPipeline":
        from .forecasting import ForecastingPipeline

        return ForecastingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def pipeline(
    task: str = "forecasting",
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
) -> "ForecastingPipeline":
    """Create a pipeline for a time series task.

    This is the recommended entry point. It returns a Pipeline object that
    handles data preparation, model building, training, and prediction.

    Args:
        task: Task type — ``'forecasting'`` (more tasks coming soon).
        model: Model name (e.g. ``'patch_tst'``, ``'transformer'``,
            ``'dlinear'``, ``'nbeats'``) or an ``AutoModel`` instance.
        lookback: Number of past steps to use as input.
        horizon: Number of future steps to predict.
        config: Optional ``AutoConfig`` for fine-grained control.
        batch_size: Batch size for training.
        normalize: Normalization — ``'minmax'``, ``'standard'``, or ``None``.
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
        early_stopping_patience: Early stopping patience.
        seed: Random seed.
        **kwargs: Passed to the model config.

    Returns:
        A Pipeline object ready for ``.fit()`` and ``.predict()``.

    Raises:
        ValueError: If the task is unknown.

    Examples:
        >>> pipe = tfts.pipeline("forecasting", model="dlinear",
        ...                      lookback=96, horizon=24)
        >>> pipe.fit(df, target_col="sales", epochs=50)
        >>> preds = pipe.predict(steps=24)
    """
    from .forecasting import ForecastingPipeline

    if task == "forecasting":
        return ForecastingPipeline(
            model=model,
            lookback=lookback,
            horizon=horizon,
            config=config,
            batch_size=batch_size,
            normalize=normalize,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            seed=seed,
            **kwargs,
        )

    raise ValueError(
        f"Unknown task '{task}'. Currently supported: 'forecasting'. "
        f"More tasks (classification, anomaly) are coming soon."
    )
