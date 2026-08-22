"""OptunaTuner — hyperparameter search for TFTS models.

Wraps Optuna to tune model configs and training hyperparameters
with a simple ``search()`` API.  Optuna is an optional dependency;
importing this module raises a helpful error if ``optuna`` is not
installed.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..models.auto_config import AutoConfig
from ..models.auto_model import AutoModel
from ..trainer import Trainer
from ..training.runtime import create_adamw

logger = logging.getLogger(__name__)


def _require_optuna() -> "optuna":  # type: ignore[name-defined]  # noqa: F821
    """Lazy-import optuna with a clear error message."""
    try:
        import optuna

        return optuna
    except ImportError:
        raise ImportError("optuna is required for OptunaTuner. " "Install it with: pip install optuna")


class OptunaTuner:
    """Hyperparameter tuner powered by Optuna.

    Given a list of model names and a parameter search space, builds and
    trains TFTS models inside an Optuna study and returns the best
    configuration.

    Args:
        train_data: Training dataset — ``(x_train, y_train)`` tuple or
            ``tf.data.Dataset``.
        valid_data: Validation dataset — same format.
        predict_sequence_length: Forecast horizon.
        metric: Metric to optimize.  ``'mse'``, ``'mae'``, or a callable
            ``f(y_true, y_pred) -> float``.
        direction: ``'minimize'`` (default) or ``'maximize'``.

    Examples:
        >>> from tfts.tuner import OptunaTuner
        >>> tuner = OptunaTuner(train_data, valid_data, predict_sequence_length=7)
        >>> best_params, best_score = tuner.search(
        ...     param_space={
        ...         "model_type": ["rnn", "dlinear"],
        ...         "learning_rate": [1e-4, 1e-2],
        ...     },
        ...     n_trials=20,
        ... )
    """

    def __init__(
        self,
        train_data: Any,
        valid_data: Any,
        predict_sequence_length: int = 1,
        metric: Union[str, Callable] = "mse",
        direction: str = "minimize",
    ) -> None:
        self.train_data = train_data
        self.valid_data = valid_data
        self.predict_sequence_length = predict_sequence_length
        self.metric = metric
        self.direction = direction

        self._study: Optional[Any] = None  # optuna.Study

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        param_space: Dict[str, Any],
        n_trials: int = 20,
        epochs: int = 10,
        verbose: int = 0,
    ) -> Tuple[Dict[str, Any], float]:
        """Run the hyperparameter search.

        Args:
            param_space: Dictionary mapping parameter names to search
                ranges::

                    {
                        "model_type": ["rnn", "dlinear"],      # categorical
                        "learning_rate": [1e-4, 1e-2],         # log-uniform
                        "hidden_size": [32, 256],              # int uniform
                        "num_layers": [1, 4],                  # int uniform
                    }

                - **list of strings** → categorical choice
                - **list of two floats** `[lo, hi]` →
                  - float log-uniform if both > 0 and lo < 1
                  - int uniform otherwise
            n_trials: Number of Optuna trials.
            epochs: Training epochs per trial.
            verbose: Keras verbosity (0 = silent).

        Returns:
            ``(best_params, best_score)`` tuple.
        """
        optuna = _require_optuna()

        # Silence optuna logs unless user wants them
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction=self.direction)
        study.optimize(
            func=lambda trial: self._objective(trial, param_space, epochs, verbose),
            n_trials=n_trials,
        )

        self._study = study
        return study.best_params, study.best_value

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best parameters found so far.

        Raises:
            RuntimeError: If :meth:`search` has not been called.
        """
        if self._study is None:
            raise RuntimeError("No search has been run yet. Call .search() first.")
        return dict(self._study.best_params)

    def get_best_score(self) -> float:
        """Return the best score found so far."""
        if self._study is None:
            raise RuntimeError("No search has been run yet. Call .search() first.")
        return float(self._study.best_value)

    def get_study(self) -> Any:
        """Return the underlying Optuna study (for advanced plotting).

        Returns ``None`` before :meth:`search` is called.
        """
        return self._study

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: Any,
        param_space: Dict[str, Any],
        epochs: int,
        verbose: int,
    ) -> float:
        """Optuna objective: build model, train, evaluate."""
        # Suggest parameters
        params = self._suggest_params(trial, param_space)

        # Extract model_type (required)
        model_type = params.pop("model_type")

        # Build config with suggested overrides
        config = AutoConfig.for_model(model_type)
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.debug(f"Config key {key!r} not found on {model_type} config, skipping")

        # Build & train model
        model = AutoModel.from_config(config, predict_sequence_length=self.predict_sequence_length)
        trainer = Trainer(model)

        # Unpack learning_rate if present — pass via optimizer
        lr = params.get("learning_rate", 1e-3)
        optimizer = _default_optimizer(lr)

        trainer.train(
            self.train_data,
            valid_dataset=self.valid_data,
            epochs=epochs,
            optimizer=optimizer,
            verbose=verbose,
        )

        # Evaluate on validation data
        metrics = trainer.evaluate(self.valid_data, metrics=[self.metric] if isinstance(self.metric, str) else None)
        score = self._extract_score(metrics)
        return score

    def _suggest_params(self, trial: Any, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert param_space into optuna suggestions."""
        params: Dict[str, Any] = {}

        for name, spec in param_space.items():
            # Categorical: list of strings
            if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], str):
                params[name] = trial.suggest_categorical(name, spec)
                continue

            # Numeric range: [lo, hi]
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                lo, hi = spec
                if isinstance(lo, float) or isinstance(hi, float):
                    # log-uniform for learning_rate style params
                    if lo > 0 and lo < 1:
                        params[name] = trial.suggest_float(name, lo, hi, log=True)
                    else:
                        params[name] = trial.suggest_float(name, lo, hi)
                else:
                    params[name] = trial.suggest_int(name, int(lo), int(hi))
                continue

            raise ValueError(
                f"Cannot infer suggestion type for param {name!r} with spec {spec!r}. "
                "Use a list of strings for categorical or [lo, hi] for numeric."
            )

        return params

    def _extract_score(self, metrics: Dict[str, float]) -> float:
        """Extract a single scalar score from the metrics dict."""
        if isinstance(self.metric, str) and self.metric in metrics:
            return float(metrics[self.metric])
        # Fallback: return first value
        if metrics:
            return float(next(iter(metrics.values())))
        raise ValueError("No metric value could be extracted from the evaluation results.")

    def __repr__(self) -> str:
        return f"OptunaTuner(metric={self.metric!r}, direction={self.direction!r})"


def _default_optimizer(lr: float):
    """Create a default optimizer with the given learning rate."""
    return create_adamw(learning_rate=lr, weight_decay=1e-4)
