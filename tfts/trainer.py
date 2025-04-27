"""tfts Trainer"""

from collections.abc import Iterable
from contextlib import nullcontext
import logging
import os
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

from .constants import TFTS_HUB_CACHE
from .models.base import BaseModel
from .training_args import TrainingArguments

__all__ = ["Trainer", "KerasTrainer", "Seq2seqKerasTrainer"]


logger = logging.getLogger(__name__)


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
        self.args = args or TrainingArguments(output_dir=TFTS_HUB_CACHE)
        self.strategy = strategy

        # with self.get_strategy_scope(strategy):
        #     self.model = self._setup_model(model)
        #     self.loss_fn = loss_fn
        #     self.metrics = metrics or []
        #     self.optimizer = optimizer or self._create_optimizer()
        #     self.lr_scheduler = lr_scheduler or self._create_lr_scheduler()
        #
        #     # Training state
        #     self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        #     if self.args.fp16:
        #         self._setup_mixed_precision()

    def evaluate(self):
        pass

    def get_train_dataloader(self):
        return

    def get_eval_dataloader(self):
        return

    def get_test_dataloader(self):
        return

    def get_learning_rates(self):
        return

    def create_accelerator_and_postprocess(self):
        return

    def get_strategy_scope(self):
        return self.strategy.scope() if self.strategy else nullcontext()

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer with specified parameters."""
        return tf.keras.optimizers.Adam(
            learning_rate=self.args.learning_rate,
            beta_1=self.args.adam_beta1,
            beta_2=self.args.adam_beta2,
            epsilon=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

    def _create_lr_scheduler(self) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Create learning rate scheduler based on arguments."""
        if self.args.lr_scheduler_type == "linear":
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.args.learning_rate,
                decay_steps=self.args.max_steps if self.args.max_steps > 0 else self.args.num_train_epochs,
                end_learning_rate=0,
                power=1.0,
            )
        return None

    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training."""
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    # def _setup_ema(self) -> None:
    #     """Configure Exponential Moving Average if enabled."""
    #     self.ema = None
    #     if self.config.use_ema:
    #         self.ema = tf.train.ExponentialMovingAverage(self.config.ema_decay)

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
        elif isinstance(x, (np.ndarray, pd.DataFrame)):
            logger.debug("Preparing single input")
            return Input(shape=x.shape[1:], name="input")
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")


class KerasTrainer(BaseTrainer):
    """Keras trainer from tf.keras"""

    def __init__(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        strategy: Optional[tf.distribute.Strategy] = None,
        run_eagerly: bool = True,
        args: Optional[TrainingArguments] = None,
        **kwargs: Dict[str, object],
    ) -> None:
        """
        Initializes the trainer with the model, loss function, optimizer, and other optional parameters.

        Args:
            model: A Keras Model or Sequential instance to train.
            loss_fn: A callable or Keras loss function. Default is MeanSquaredError.
            optimizer: A Keras optimizer instance. Default is Adam with learning rate 0.003.
            lr_scheduler: Optional learning rate scheduler.
            strategy: Optional distribution strategy for multi-GPU or multi-node training.
            run_eagerly: Whether to run eagerly. Default is True.
            **kwargs: Additional arguments that are passed to the instance as attributes.
        """
        super().__init__(model, args, strategy, **kwargs)
        self.model = model
        self.config = model.config if hasattr(model, "config") else None
        self.run_eagerly = run_eagerly

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_dataset: Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
        valid_dataset: Optional[Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]] = None,
        loss_fn: Union[Callable, tf.keras.losses.Loss, str] = "mse",
        optimizer: Union[tf.keras.optimizers.Optimizer, str] = "adam",
        lr_scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        epochs: int = 10,
        batch_size: int = 64,
        steps_per_epoch: Optional[int] = None,
        metrics: Optional[Union[List[tf.keras.metrics.Metric], List[str]]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: int = 1,
        **kwargs: Dict[str, object],
    ) -> tf.keras.callbacks.History:
        """
        Trains the model on the provided dataset.

        Args:
            train_dataset: A tf.data.Dataset or list/tuple of tensors (x_train, y_train).
            valid_dataset: A tf.data.Dataset or list/tuple of tensors (x_valid, y_valid), optional.
            epochs: Number of epochs to train for. Default is 10.
            batch_size: Number of samples per batch. Default is 64.
            steps_per_epoch: Number of steps per epoch. Optional.
            metrics:  List of metrics for monitoring during training. Optional.
            callbacks: List of keras callbacks during training. Optional.
            verbose: Verbosity level. Default is 1.
            **kwargs: Additional keyword arguments for callbacks.

        Returns:
            A History object containing training logs.
        """

        if not callbacks:
            callbacks: List[tf.keras.callbacks.Callback] = []

        inputs = self.get_inputs(train_dataset)

        with self.get_strategy_scope():
            if lr_scheduler:
                callbacks.append(lr_scheduler)

            if not isinstance(self.model, tf.keras.Model):
                if "build_model" not in dir(self.model):
                    raise TypeError("Trainer model should either be `tf.keras.Model` or has `build_model()` method")
                self.model = self.model.build_model(inputs=inputs)

            self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics, run_eagerly=self.run_eagerly)

            trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            tf.print(f"Trainable parameters: {trainable_params}")

            if isinstance(train_dataset, (list, tuple)):
                x_train, y_train = train_dataset

                history = self.model.fit(
                    x_train,
                    y_train,
                    validation_data=valid_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                    callbacks=callbacks,
                )
            else:
                history = self.model.fit(
                    train_dataset,
                    validation_data=valid_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                    callbacks=callbacks,
                )
        return history

    def fit(self, **params):
        return self.train(**params)

    def predict(self, x_test: tf.Tensor) -> tf.Tensor:
        y_test_pred = self.model(x_test)
        return y_test_pred

    def get_model(self) -> tf.keras.Model:
        return self.model

    def save_model(self, model_dir: str, save_weights_only: bool = True, checkpoint_dir: Optional[str] = None):
        # save the model, checkpoint_dir if you use Checkpoint callback to save your best weights
        if checkpoint_dir is not None:
            logger.info("checkpoint Loaded", checkpoint_dir)
            self.model.load_weights(checkpoint_dir)
        else:
            logger.info("No checkpoint Loaded")

        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, "model.h5"))
        if self.config is not None:
            self.config.to_json(os.path.join(model_dir, "config.json"))
        logger.info("protobuf model successfully saved in {}".format(model_dir))

        if not save_weights_only:
            self.model.save_weights(f"{model_dir}.ckpt")
            logger.info(f"model weights successfully saved in {model_dir}.ckpt")
        return

    def plot(self, history, true: np.ndarray, pred: np.ndarray):
        import matplotlib.pyplot as plt

        train_length = history.shape[1]
        pred_length = true.shape[1]
        example = np.random.choice(range(history.shape[0]))

        plt.plot(range(train_length), history[example, :, 0], label="History")
        plt.plot(range(train_length, train_length + pred_length), true[example, :, 0], label="True")
        plt.plot(range(train_length, train_length + pred_length), pred[example, :, 0], label="Predicted")
        plt.legend()


class Seq2seqKerasTrainer(KerasTrainer):
    """As the transformers forum mentioned: https://discuss.huggingface.co/t/trainer-vs-seq2seqtrainer/3145/2
    Seq2SeqTrainer is mostly about predict_with_generate."""

    def __init__(self, *args, **kwargs):
        super(Seq2seqKerasTrainer, self).__init__(*args, **kwargs)


class Trainer(object):
    """Custom trainer for tensorflow with support for CPU, GPU, and multi-GPU."""

    def __init__(
        self,
        model,
        strategy: Optional[tf.distribute.Strategy] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model = model
        self.strategy = strategy

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_loader: Union[tf.data.Dataset, Generator],
        valid_loader: Union[tf.data.Dataset, Generator, None] = None,
        loss_fn: Union[Callable] = tf.keras.losses.MeanSquaredError(),
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.003),
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
    ) -> None:
        """
        Trains the model using the provided data loaders.

        Parameters
        ----------
        train_loader : Union[tf.data.Dataset, Generator]
            The training data loader, which can be a `tf.data.Dataset` or a Python generator.
        valid_loader : Union[tf.data.Dataset, Generator, None], optional
            The validation data loader, by default None.
        epochs : int, optional
            The number of epochs to train the model, by default 10.
        learning_rate : float, optional
            The initial learning rate for the optimizer, by default 3e-4.
        verbose : int, optional
            The verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch), by default 1.
        eval_metric : Union[Callable, List[Callable], None], optional
            The evaluation metric(s) to use for validation, by default None.
        model_dir : Optional[str], optional
            The directory to save the model weights, by default "../weights".
        use_ema : bool, optional
            Whether to use exponential moving average (EMA) for the model weights, by default False.
        stop_no_improve_epochs : Optional[int], optional
            If provided, training will stop if the validation metric does not improve for the specified
            number of epochs, by default None.
        max_grad_norm : float, optional
            the max gradient while backprop.
        transform : Optional[Callable], optional
            A function to transform the data before feeding it to the model, by default None.
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric if isinstance(eval_metric, Iterable) else [eval_metric]
        self.use_ema = use_ema
        self.transform = transform
        self.max_grad_norm = max_grad_norm
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        if use_ema:
            self.ema = tf.train.ExponentialMovingAverage(0.9).apply(self.model.trainable_variables)

        if model_dir is None:
            model_dir = TFTS_HUB_CACHE

        if stop_no_improve_epochs is not None:
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

        for epoch in range(epochs):
            train_loss, train_scores = self.train_loop(train_loader)
            log_str = f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}"

            if valid_loader is not None:
                valid_loss, valid_scores = self.valid_loop(valid_loader)
                log_str += f", Valid Loss: {valid_loss:.4f}"
                log_str + ",".join([" Valid Metrics{}: {:.4f}".format(i, me) for i, me in enumerate(valid_scores)])

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

        # self.export_model(model_dir, only_pb=True)  # save the model

    def fit(self, **params):
        return self.train(**params)

    def train_loop(self, train_loader):
        train_loss: float = 0.0
        y_trues, y_preds = [], []

        for step, (x_train, y_train) in enumerate(train_loader):
            y_pred, step_loss = self.train_step(x_train, y_train)
            train_loss += step_loss
            y_preds.append(y_pred)
            y_trues.append(y_train)

        scores = []
        if self.eval_metric is not None:
            y_preds = tf.concat(y_preds, axis=0)
            y_trues = tf.concat(y_trues, axis=0)

            for metric in self.eval_metric:
                scores.append(metric(y_trues, y_preds))
        return train_loss / (step + 1), scores

    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)
            loss = self.loss_fn(y_train, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm)) for grad in gradients]
        _ = self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.global_step)
            self.optimizer.lr.assign(lr)
        else:
            lr = self.learning_rate
        self.optimizer.lr.assign(lr)
        self.global_step.assign_add(1)
        # logger.info('Step: {}, Loss: {}'.format(self.global_step.numpy(), loss))
        return y_pred, loss

    def valid_loop(self, valid_loader):
        valid_loss: float = 0.0
        y_valid_trues, y_valid_preds = [], []

        for valid_step, (x_valid, y_valid) in enumerate(valid_loader):
            y_valid_pred, valid_step_loss = self.valid_step(x_valid, y_valid)
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

    def valid_step(self, x_valid, y_valid):

        y_valid_pred = self.model(x_valid, training=False)
        valid_loss = self.loss_fn(y_valid, y_valid_pred)
        return y_valid_pred, valid_loss

    def predict(self, test_loader):
        y_test_trues, y_test_preds = [], []
        for x_test, y_test in test_loader:
            y_test_pred = self.model(x_test, training=False)
            y_test_preds.append(y_test_pred)
            y_test_trues.append(y_test)

        y_test_trues = tf.concat(y_test_trues, axis=0)
        y_test_preds = tf.concat(y_test_preds, axis=0)
        return tf.squeeze(y_test_trues, axis=-1), y_test_preds

    def save_model(self, model_dir, only_pb=True):
        # save the model
        tf.saved_model.save(self.model, model_dir)
        logger.info(f"Protobuf model successfully saved in {model_dir}")

        if not only_pb:
            self.model.save_weights(f"{model_dir}.ckpt")
            logger.info(f"Model weights successfully saved in {model_dir}.ckpt")
