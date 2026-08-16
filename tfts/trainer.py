from collections.abc import Iterable
from contextlib import nullcontext
import logging
import os
import random
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

from .constants import CONFIG_NAME, TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, TFTS_HOME, TFTS_HUB_CACHE
from .models.base import BaseModel
from .training.runtime import configure_precision, create_distribution_strategy
from .training_args import TrainingArguments

__all__ = ["Trainer", "KerasTrainer", "EagerTrainer", "Seq2seqKerasTrainer", "set_seed"]


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


class BaseTrainer(object):
    """Trainer for pipeline"""

    def __init__(
        self,
        model: Union[tf.keras.Model, "BaseModel"],
        args: Optional[TrainingArguments] = None,
        strategy: Optional[tf.distribute.Strategy] = None,
        **kwargs,
    ):
        self.model = model
        self.config = model.config if hasattr(model, "config") else None
        self.args = args or TrainingArguments(output_dir=TFTS_HUB_CACHE)
        self.strategy = strategy or create_distribution_strategy(self.args)

    def evaluate(self) -> None:
        pass

    def get_train_dataloader(self) -> Any:
        return

    def get_eval_dataloader(self) -> Any:
        return

    def get_test_dataloader(self) -> Any:
        return

    def get_learning_rates(self) -> Any:
        return

    def create_accelerator_and_postprocess(self) -> Any:
        return

    @staticmethod
    def get_distribution_strategy() -> tf.distribute.Strategy:
        return create_distribution_strategy()

    def get_strategy_scope(self) -> Union[tf.distribute.Strategy.scope, nullcontext]:
        return self.strategy.scope() if self.strategy else nullcontext()

    def _create_optimizer(
        self,
        learning_rate: Optional[Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]] = None,
    ) -> tf.keras.optimizers.Optimizer:
        """Create optimizer with specified parameters."""
        learning_rate = learning_rate if learning_rate is not None else self.args.learning_rate
        # tf.keras.optimizers.Adam does not support weight_decay directly.
        # Use AdamW if available, otherwise fall back to standard Adam.
        try:
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                beta_1=self.args.adam_beta1,
                beta_2=self.args.adam_beta2,
                epsilon=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        except AttributeError:
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=self.args.adam_beta1,
                beta_2=self.args.adam_beta2,
                epsilon=self.args.adam_epsilon,
            )

    def _create_lr_scheduler(self) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Create learning rate scheduler based on arguments."""
        decay_steps = self.args.max_steps if self.args.max_steps > 0 else self.args.num_train_epochs
        decay_steps = max(1, int(decay_steps))
        if self.args.lr_scheduler_type == "linear":
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.args.learning_rate,
                decay_steps=decay_steps,
                end_learning_rate=0,
                power=1.0,
            )
        return None

    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training."""
        configure_precision(self.args)

    def get_inputs(self, train_dataset):
        if isinstance(train_dataset, tf.data.Dataset):
            # choose the first batch
            x = next(iter(train_dataset.take(1).as_numpy_iterator()))[0]
            inputs = self._prepare_inputs_for_model(x)

        elif isinstance(train_dataset, tf.keras.utils.Sequence):
            x, _ = train_dataset[0]
            inputs = self._prepare_inputs_for_model(x)

        elif isinstance(train_dataset, (list, tuple)):
            x = train_dataset[0]
            inputs = self._prepare_inputs_for_model(x)
        else:
            raise ValueError("Unsupported dataset type. Expected tf.data.Dataset, keras.utils.Sequence, or list/tuple.")
        return inputs

    def _keras_model_for_saving(self) -> tf.keras.Model:
        if isinstance(self.model, tf.keras.Model):
            return self.model
        if isinstance(self.model, BaseModel):
            model = self.model.model
            if isinstance(model, tf.keras.Model):
                return model
        raise ValueError(
            "Model weights cannot be saved before the model is built. "
            "Call `train(...)`, or build the model with input shapes before `save_model(...)`."
        )

    def _prepare_inputs_for_model(
        self, x: Union[np.ndarray, pd.DataFrame]
    ) -> Union[Dict[str, tf.keras.layers.Input], List[tf.keras.layers.Input], tf.keras.layers.Input]:
        """
        Prepares the input layer(s) based on the shape of the provided data.

        Args:
            x: Input data (either a NumPy array or a Pandas DataFrame).

        Returns:
            The corresponding Keras Input layers.
        """
        if isinstance(x, dict):
            logger.debug("Preparing inputs from dict")
            return {key: Input(shape=item.shape[1:], name=key) for key, item in x.items()}
        elif isinstance(x, (list, tuple)):
            logger.debug("Preparing inputs from list or tuple")
            return [Input(shape=item.shape[1:], name=f"input_{i}") for i, item in enumerate(x)]
        else:
            logger.debug("Preparing single input")
            return Input(shape=x.shape[1:], name="input")

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else TFTS_HOME
        logger.info(f"Saving model checkpoint to {output_dir}")
        # save_model = self.model.model if hasattr(self.model, "model") else self.model
        # self.model.save_pretrained(output_dir)

        # model save (due to after build_model, the model will be replaced to a tf.keras.model)
        save_directory = output_dir
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        keras_model = self._keras_model_for_saving()

        os.makedirs(save_directory, exist_ok=True)
        # Use model_type from config if available, otherwise derive from class name
        name = self.model.__class__.__name__
        architecture = getattr(self.config, "model_type", name)
        if self.config is not None:
            self.config.architectures = [architecture]
            self.config.save_pretrained(save_directory)

        weights_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)  # Or the appropriate extension

        keras_model.save_weights(weights_file)
        logging.info(f"Model weights successfully saved in {weights_file}")

    @property
    def global_batch_size(self):
        return self.args.per_device_train_batch_size * self.strategy.num_replicas_in_sync


class Trainer(BaseTrainer):
    """Unified Trainer for time series tasks.

    Automatically selects loss, optimizer, and metrics based on task type.
    Supports both eager and compiled (model.fit) training.

    Examples:
        >>> from tfts import AutoModel, AutoConfig, Trainer
        >>> config = AutoConfig.for_model("transformer")
        >>> model = AutoModel.from_config(config, predict_sequence_length=12)
        >>> trainer = Trainer(model)
        >>> trainer.train(train_dataset, valid_dataset, epochs=50)
    """

    def __init__(
        self,
        model: Union[tf.keras.Model, "BaseModel"],
        strategy: Optional[tf.distribute.Strategy] = None,
        args: Optional[TrainingArguments] = None,
        **kwargs: Dict[str, object],
    ) -> None:
        super().__init__(model, args, strategy, **kwargs)
        self.model = model
        self.config = model.config if hasattr(model, "config") else None
        self._task: str = "forecasting"  # 'forecasting', 'classification', 'anomaly'

        for key, value in kwargs.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
        valid_dataset: Optional[Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]] = None,
        loss_fn: Union[Callable, tf.keras.losses.Loss, str, None] = None,
        optimizer: Union[tf.keras.optimizers.Optimizer, str, Dict, None] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        metrics: Optional[Union[List[tf.keras.metrics.Metric], List[str]]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        reduce_lr_patience: Optional[int] = None,
        run_eagerly: bool = True,
        verbose: int = 1,
        **kwargs: Dict[str, object],
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            train_dataset: Training data — tf.data.Dataset or (x, y) tuple.
            valid_dataset: Validation data (optional).
            loss_fn: Loss function. Auto-selected if None.
            optimizer: Optimizer. Defaults to Adam with config learning rate.
            epochs: Number of training epochs.
            batch_size: Samples per batch.
            steps_per_epoch: Steps per epoch (for infinite datasets).
            metrics: Keras metrics to track. Auto-selected if None.
            callbacks: Keras callbacks. Built-in callbacks are added automatically.
            early_stopping_patience: If set, adds EarlyStopping callback.
            checkpoint_dir: If set, saves best model weights.
            reduce_lr_patience: If set, adds ReduceLROnPlateau callback.
            run_eagerly: Run eagerly (True) or with tf.function (False).
            verbose: 0=silent, 1=progress bar, 2=one line per epoch.

        Returns:
            A History object.
        """
        callbacks = list(callbacks) if callbacks else []
        configure_precision(self.args)
        epochs = int(epochs if epochs is not None else self.args.num_train_epochs)
        # `batch_size` is a per-device value, consistent with `per_device_train_batch_size`
        # and `global_batch_size`. `model.fit` expects a *global* batch that is then split
        # across the strategy replicas, so scale it here on multi-device setups.
        if batch_size is None:
            batch_size = int(self.args.per_device_train_batch_size)
        else:
            batch_size = int(batch_size)
        batch_size *= self.strategy.num_replicas_in_sync if self.strategy is not None else 1

        # Auto-build callbacks
        callbacks += self._build_callbacks(
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir,
            reduce_lr_patience=reduce_lr_patience,
        )

        with self.get_strategy_scope():
            # Auto-select loss & optimizer
            if loss_fn is None:
                loss_fn = self._default_loss()
            if optimizer is None:
                optimizer = self._create_optimizer(learning_rate=self._create_lr_scheduler())
            if metrics is None:
                metrics = self._default_metrics()

            if isinstance(optimizer, (str, dict)):
                optimizer = tf.keras.optimizers.get(optimizer)

            # Build model if needed
            if not isinstance(self.model, tf.keras.Model):
                inputs = self.get_inputs(train_dataset)
                if "build_model" not in dir(self.model):
                    raise TypeError("Trainer model must be `tf.keras.Model` or have `build_model()`")
                self.model = self.model.build_model(inputs=inputs)
            elif self.strategy is not None and self.strategy.num_replicas_in_sync > 1:
                # A pre-built Keras model has variables that were created outside this
                # strategy scope. TensorFlow refuses to mix scopes (colocate_vars_with),
                # so re-create the same architecture inside the scope and copy weights.
                rebuilt = tf.keras.models.clone_model(self.model)
                rebuilt.build(self.model.input_shape)
                rebuilt.set_weights(self.model.get_weights())
                self.model = rebuilt

            compile_kwargs = {
                "loss": loss_fn,
                "optimizer": optimizer,
                "metrics": metrics,
                "run_eagerly": run_eagerly,
            }
            if self.args.jit_compile:
                compile_kwargs["jit_compile"] = True
            self.model.compile(**compile_kwargs)

            trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]))
            logger.info(f"Trainable parameters: {trainable_params:,}")

            # Normalize raw numpy/list inputs to a globally-batched tf.data.Dataset.
            # Feeding numpy arrays to `model.fit` under a real distribution strategy
            # triggers "Mixing different tf.distribute.Strategy objects" in Keras 3,
            # and a too-small batch splits unevenly across replicas. Batching here to
            # the global (replica-scaled) batch keeps behavior identical on a single
            # device (num_replicas == 1) and correct on multi-GPU.
            if isinstance(train_dataset, (list, tuple)):
                x_train, y_train = train_dataset
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
            if isinstance(valid_dataset, (list, tuple)):
                x_valid, y_valid = valid_dataset
                valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

            history = self.model.fit(
                train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                batch_size=None,
                verbose=verbose,
                callbacks=callbacks,
            )
        return history

    def fit(self, **params):
        """Alias for train()."""
        return self.train(**params)

    def evaluate(
        self,
        dataset: Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            dataset: Evaluation data.
            metrics: Metric names (e.g. ['mse', 'mae']). Uses built-in metrics if None.

        Returns:
            Dictionary of metric_name -> value.
        """
        from .metrics import evaluate as compute_metrics

        if isinstance(dataset, (list, tuple)):
            x, y_true = dataset
            y_pred = self.model(x, training=False)
        elif isinstance(dataset, tf.data.Dataset):
            y_true_list, y_pred_list = [], []
            for x_batch, y_batch in dataset:
                y_pred_list.append(self.model(x_batch, training=False))
                y_true_list.append(y_batch)
            y_true = tf.concat(y_true_list, axis=0)
            y_pred = tf.concat(y_pred_list, axis=0)
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        return compute_metrics(y_true, y_pred, metrics=metrics or "all")

    def predict(self, x: Union[tf.Tensor, np.ndarray, tf.data.Dataset]) -> np.ndarray:
        """Make predictions.

        Args:
            x: Single input tensor/array or a tf.data.Dataset.

        Returns:
            Numpy array of predictions.
        """
        if isinstance(x, tf.data.Dataset):
            preds = []
            for batch in x:
                inp = batch[0] if isinstance(batch, (tuple, list)) else batch
                preds.append(self.model(inp, training=False))
            return tf.concat(preds, axis=0).numpy()
        return self.model(x, training=False).numpy()

    def get_model(self) -> tf.keras.Model:
        """Return the underlying Keras model."""
        return self.model

    def save_model(self, output_dir: Optional[str] = None):
        """Save model weights and config."""
        if hasattr(self.strategy, "cluster_resolver") and self.strategy.cluster_resolver:
            if self.strategy.cluster_resolver.task_type != "chief":
                return
        output_dir = TFTS_HOME if output_dir is None else output_dir
        self._save(output_dir)

    @staticmethod
    def plot(history: np.ndarray, true: np.ndarray, pred: np.ndarray):
        """Quick plot of history, ground truth, and predictions."""
        import matplotlib.pyplot as plt

        train_length = history.shape[1]
        pred_length = true.shape[1]
        example = np.random.choice(range(history.shape[0]))

        plt.plot(range(train_length), history[example, :, 0], label="History")
        plt.plot(range(train_length, train_length + pred_length), true[example, :, 0], label="True")
        plt.plot(range(train_length, train_length + pred_length), pred[example, :, 0], label="Predicted")
        plt.legend()

    # ------------------------------------------------------------------
    # Auto-config helpers
    # ------------------------------------------------------------------

    def _default_loss(self) -> tf.keras.losses.Loss:
        """Return a sensible default loss for the current task."""
        if self._task == "classification":
            return tf.keras.losses.SparseCategoricalCrossentropy()
        return tf.keras.losses.MeanSquaredError()

    def _default_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Return a sensible default optimizer."""
        return self._create_optimizer(learning_rate=self._create_lr_scheduler())

    def _default_metrics(self) -> List[str]:
        """Return default metrics for monitoring."""
        if self._task == "classification":
            return ["accuracy"]
        return ["mae"]

    @staticmethod
    def _build_callbacks(
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        reduce_lr_patience: Optional[int] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        """Build standard Keras callbacks from simple parameters."""
        callbacks: List[tf.keras.callbacks.Callback] = []

        if early_stopping_patience is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, "best_weights.weights.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                )
            )
        if reduce_lr_patience is not None:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    patience=reduce_lr_patience,
                    factor=0.5,
                    min_lr=1e-7,
                )
            )
        return callbacks


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

KerasTrainer = Trainer  # legacy alias


class Seq2seqKerasTrainer(Trainer):
    """Seq2SeqTrainer — supports predict_with_generate.

    See: https://discuss.huggingface.co/t/trainer-vs-seq2seqtrainer/3145/2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EagerTrainer(object):
    """Low-level custom training loop trainer (legacy).

    Use ``Trainer`` for the standard high-level API. This class provides
    manual gradient-tape-based training for users who need full control.
    """

    def __init__(
        self,
        model,
        strategy: Optional[tf.distribute.Strategy] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model = model
        self.strategy = strategy or tf.distribute.get_strategy()

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_loader: Union[tf.data.Dataset, Generator],
        valid_loader: Union[tf.data.Dataset, Generator, None] = None,
        loss_fn: Union[Callable] = tf.keras.losses.MeanSquaredError(),
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        lr_scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        epochs: int = 10,
        learning_rate: float = 3e-4,
        verbose: int = 1,
        eval_metric: Union[Callable, List[Callable], None] = None,
        model_dir: Optional[str] = None,
        use_ema: bool = False,
        stop_no_improve_epochs: Optional[int] = None,
        max_grad_norm: float = 5.0,
        transform: Optional[Callable] = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Train with manual gradient tape loop."""
        self.loss_fn = loss_fn
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(0.003)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        if eval_metric is None:
            self.eval_metric = []
        else:
            self.eval_metric = eval_metric if isinstance(eval_metric, Iterable) else [eval_metric]
        self.use_ema = use_ema
        self.transform = transform
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        if model_dir is None:
            model_dir = TFTS_HUB_CACHE

        no_improve_epochs: int = 0
        best_metric: float = float("inf")

        if not isinstance(self.model, tf.keras.Model):
            if "build_model" not in dir(self.model):
                raise TypeError("Trainer model should either be tf.keras.Model or has the build_model method")
            x = list(train_loader.take(1).as_numpy_iterator())[0][0]
            if isinstance(x, dict):
                inputs = {key: Input(item.shape[1:]) for key, item in x.items()}
            else:
                inputs = Input(x.shape[1:])
            self.model = self.model.build_model(inputs=inputs)

        if use_ema:
            try:
                self.ema = tf.train.ExponentialMovingAverage(0.9).apply(self.model.trainable_variables)
            except Exception as e:
                logger.warning(f"Failed to apply EMA: {e}")
                self.ema = None

        for epoch in range(epochs):
            train_loss, train_scores = self._train_loop(train_loader)
            log_str = f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}"

            if valid_loader is not None:
                valid_loss, valid_scores = self._valid_loop(valid_loader)
                log_str += f", Valid Loss: {valid_loss:.4f}"
                log_str += ",".join([f" Valid Metrics{i}: {me:.4f}" for i, me in enumerate(valid_scores)])

                if (stop_no_improve_epochs is not None) and (eval_metric is not None):
                    if valid_scores[0] >= best_metric:
                        best_metric = valid_scores[0]
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                    if no_improve_epochs >= stop_no_improve_epochs:
                        logger.info("Tried the best, no improved and stop training")
                        break

            logger.info(log_str)

    def fit(self, **params):
        return self.train(**params)

    def _train_loop(self, train_loader: Any) -> tuple[float, list[Any]]:
        train_loss: float = 0.0
        y_trues, y_preds = [], []
        accum_grads: Optional[List[Optional[tf.Tensor]]] = None
        accum_steps = self.gradient_accumulation_steps
        step = 0

        for step, (x_train, y_train) in enumerate(train_loader):
            with tf.GradientTape() as tape:
                y_pred = self.model(x_train, training=True)
                loss = self.loss_fn(y_train, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)

            # Accumulate gradients over micro-batches before applying an update.
            if accum_grads is None:
                accum_grads = [tf.identity(g) if g is not None else None for g in grads]
            else:
                accum_grads = [(a if g is None else g if a is None else a + g) for a, g in zip(accum_grads, grads)]

            train_loss += float(loss)
            y_preds.append(y_pred)
            y_trues.append(y_train)

            if (step + 1) % accum_steps == 0:
                self._apply_gradients(accum_grads, accum_steps)
                accum_grads = None

        # Flush any remaining accumulated micro-batches.
        if accum_grads is not None:
            self._apply_gradients(accum_grads, accum_steps)

        scores = []
        if self.eval_metric:
            y_preds = tf.concat(y_preds, axis=0)
            y_trues = tf.concat(y_trues, axis=0)
            for metric in self.eval_metric:
                scores.append(metric(y_trues, y_preds))
        return train_loss / (step + 1), scores

    def _apply_gradients(self, accum_grads: List[Optional[tf.Tensor]], accum_steps: int) -> None:
        """Normalize and apply accumulated gradients."""
        scaled = [
            tf.clip_by_value(g / accum_steps, -self.max_grad_norm, self.max_grad_norm) if g is not None else None
            for g in accum_grads
        ]
        grads_and_vars = [(g, v) for g, v in zip(scaled, self.model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.global_step)
            self.optimizer.learning_rate.assign(lr)
        else:
            lr = self.learning_rate
        self.optimizer.learning_rate.assign(lr)
        self.global_step.assign_add(1)

    def _valid_loop(self, valid_loader: Any) -> tuple[float, list[Any]]:
        valid_loss: float = 0.0
        y_valid_trues, y_valid_preds = [], []
        for valid_step, (x_valid, y_valid) in enumerate(valid_loader):
            y_valid_pred, valid_step_loss = self._valid_step(x_valid, y_valid)
            valid_loss += valid_step_loss
            y_valid_trues.append(y_valid)
            y_valid_preds.append(y_valid_pred)
        valid_scores = []
        if self.eval_metric:
            y_valid_preds = tf.concat(y_valid_preds, axis=0)
            y_valid_trues = tf.concat(y_valid_trues, axis=0)
            for metric in self.eval_metric:
                valid_scores.append(metric(y_valid_trues, y_valid_preds))
        return valid_loss / (valid_step + 1), valid_scores

    def _valid_step(self, x_valid: tf.Tensor, y_valid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        y_valid_pred = self.model(x_valid, training=False)
        valid_loss = self.loss_fn(y_valid, y_valid_pred)
        return y_valid_pred, valid_loss

    def predict(self, test_loader: Any) -> tuple[tf.Tensor, tf.Tensor]:
        y_test_trues, y_test_preds = [], []
        for x_test, y_test in test_loader:
            y_test_pred = self.model(x_test, training=False)
            y_test_preds.append(y_test_pred)
            y_test_trues.append(y_test)
        y_test_trues = tf.concat(y_test_trues, axis=0)
        y_test_preds = tf.concat(y_test_preds, axis=0)
        return tf.squeeze(y_test_trues, axis=-1), y_test_preds

    def save_model(self, model_dir, only_pb=True):
        if not model_dir.endswith(".keras"):
            model_dir = f"{model_dir}.keras"
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        self.model.save(model_dir)
        logger.info(f"Model successfully saved in {model_dir}")
        if not only_pb:
            self.model.save_weights(f"{model_dir}.ckpt")
            logger.info(f"Model weights successfully saved in {model_dir}.ckpt")
