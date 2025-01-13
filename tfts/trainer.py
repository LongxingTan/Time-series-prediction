"""tfts Trainer"""

from collections.abc import Iterable
import logging
import os
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

from .constants import TFTS_HUB_CACHE

__all__ = ["Trainer", "KerasTrainer", "Seq2seqKerasTrainer"]


logger = logging.getLogger(__name__)


class Trainer(object):
    """Custom trainer for tensorflow with support for CPU, GPU, and multi-GPU."""

    def __init__(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        loss_fn: Union[Callable] = tf.keras.losses.MeanSquaredError(),
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.003),
        lr_scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        strategy: Optional[tf.distribute.Strategy] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.strategy = strategy

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_loader: Union[tf.data.Dataset, Generator],
        valid_loader: Union[tf.data.Dataset, Generator, None] = None,
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
        # valid step for one batch
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

    def export_model(self, model_dir, only_pb=True):
        # save the model
        tf.saved_model.save(self.model, model_dir)
        logger.info(f"Protobuf model successfully saved in {model_dir}")

        if not only_pb:
            self.model.save_weights(f"{model_dir}.ckpt")
            logger.info(f"Model weights successfully saved in {model_dir}.ckpt")


class KerasTrainer(object):
    """Keras trainer from tf.keras"""

    def __init__(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        loss_fn: Union[Callable, tf.keras.losses.Loss] = tf.keras.losses.MeanSquaredError(),
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(0.003),
        lr_scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        strategy: Optional[tf.distribute.Strategy] = None,
        run_eagerly: bool = True,
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
        self.model = model
        self.config = model.config if hasattr(model, "config") else None
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.strategy = strategy
        self.run_eagerly = run_eagerly

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_dataset: Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
        valid_dataset: Optional[Union[tf.data.Dataset, List[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]] = None,
        epochs: int = 10,
        batch_size: int = 64,
        steps_per_epoch: Optional[int] = None,
        callback_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        early_stopping: Optional[tf.keras.callbacks.Callback] = None,
        checkpoint: Optional[tf.keras.callbacks.Callback] = None,
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
            callback_metrics: List of metrics for monitoring during training. Optional.
            early_stopping: Early stopping callback, optional.
            checkpoint: Checkpoint callback, optional.
            verbose: Verbosity level. Default is 1.
            **kwargs: Additional keyword arguments for callbacks.

        Returns:
            A History object containing training logs.
        """

        callbacks: List[tf.keras.callbacks.Callback] = []
        if early_stopping is not None:
            callbacks.append(early_stopping)
        if checkpoint is not None:
            callbacks.append(checkpoint)
        if "callbacks" in kwargs:
            callbacks += kwargs.get("callbacks")
            logger.info("Callbacks: ", callbacks)

        # if self.strategy is None:
        #     self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        # else:
        #     train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        #     if valid_dataset is not None:
        #         valid_dataset = self.strategy.experimental_distribute_dataset(valid_dataset)

        # with self.strategy.scope():
        if not isinstance(self.model, tf.keras.Model):
            if "build_model" not in dir(self.model):
                raise TypeError("Trainer model should either be `tf.keras.Model` or has `build_model()` method")
            if isinstance(train_dataset, tf.data.Dataset):
                # first 0 choose the batch, second 0 choose the x
                x = list(train_dataset.take(1).as_numpy_iterator())[0][0]
                inputs = self._prepare_inputs(x)

            elif isinstance(train_dataset, (list, tuple)):
                # for encoder only model, single array inputs
                if isinstance(train_dataset[0], (np.ndarray, pd.DataFrame)):
                    inputs = Input(train_dataset[0].shape[1:])
                # for encoder decoder model, 3 item of array as inputs
                elif isinstance(train_dataset[0], (list, tuple)):
                    inputs = [Input(item.shape[1:]) for item in train_dataset[0]]
                # for encoder decoder model, 3 item dict as inputs
                elif isinstance(train_dataset[0], dict):
                    inputs = {key: Input(item.shape[1:]) for key, item in train_dataset[0].items()}
            else:
                raise ValueError("tfts inputs should be either tf.data instance or 3d array list/tuple")

            if hasattr(self.model, "build_model"):
                self.model = self.model.build_model(inputs=inputs)
            else:
                raise ValueError("tfts model should have build_model func")

        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        logger.info(f"Trainable parameters: {trainable_params}")

        self.model.compile(
            loss=self.loss_fn, optimizer=self.optimizer, metrics=callback_metrics, run_eagerly=self.run_eagerly
        )
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

    def predict(self, x_test: tf.Tensor, batch_size: int = 1) -> tf.Tensor:
        y_test_pred = self.model(x_test, batch_size=batch_size)
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

        self.model.save(model_dir)
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

    def _prepare_inputs(
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
            return {key: Input(item.shape[1:]) for key, item in x.items()}
        else:
            return Input(x.shape[1:])


class Seq2seqKerasTrainer(KerasTrainer):
    """As the transformers forum mentioned: https://discuss.huggingface.co/t/trainer-vs-seq2seqtrainer/3145/2
    Seq2SeqTrainer is mostly about predict_with_generate."""

    def __init__(self, *args, **kwargs):
        super(Seq2seqKerasTrainer, self).__init__(*args, **kwargs)
