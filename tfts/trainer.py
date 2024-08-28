"""tfts Trainer"""

from collections.abc import Iterable
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

__all__ = ["Trainer", "KerasTrainer", "Seq2seqKerasTrainer"]

logger = logging.getLogger(__name__)


class Trainer(object):
    """Custom trainer for tensorflow"""

    def __init__(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        loss_fn: Union[Callable] = tf.keras.losses.MeanSquaredError(),
        optimizer: tf.keras.optimizers = tf.keras.optimizers.legacy.Adam(0.003),
        lr_scheduler: Optional[tf.keras.optimizers.Optimizer] = None,
        strategy: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        **kwargs: Dict[str, Any]
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
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        verbose: int = 1,
        eval_metric: Union[Callable, List[Callable], None] = None,
        model_dir: Optional[str] = None,
        use_ema: bool = False,
        stop_no_improve_epochs: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """train function

        Parameters
        ----------
        train_loader : _type_
            tf.data.Dataset instance
        valid_loader : _type_
            tf.data.Dataset instance
        epochs : int, optional
            _description_, by default 10
        batch_size : int, optional
            _description_, by default 8
        learning_rate : _type_, optional
            _description_, by default 3e-4
        verbose : int, optional
            _description_, by default 1
        eval_metric : tuple, optional
            _description_, by default ()
        model_dir : _type_, optional
            _description_, by default None
        use_ema : bool, optional
            _description_, by default False
        stop_no_improve_epochs : _type_, optional
            if None, no early stop; otherwise, training will stop after no_improve_epochs, by default None
        transform : _type_, optional
            _description_, by default None
        """
        self.learning_rate = learning_rate
        if eval_metric:
            if isinstance(eval_metric, Iterable):
                self.eval_metric = eval_metric
            else:
                self.eval_metric = [eval_metric]
        self.use_ema = use_ema
        self.transform = transform
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        if use_ema:
            self.ema = tf.train.ExponentialMovingAverage(0.9).apply(self.model.trainable_variables)

        if model_dir is None:
            model_dir = "../weights"

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
            log_str = "Epoch: {}, Train Loss: {:.4f}".format(epoch + 1, train_loss)

            if valid_loader is not None:
                valid_loss, valid_scores = self.valid_loop(valid_loader)
                log_str += ", Valid Loss: {:.4f}".format(valid_loss)
                log_str + ",".join([" Valid Metrics{}: {:.4f}".format(i, me) for i, me in enumerate(valid_scores)])

                if (stop_no_improve_epochs is not None) & (eval_metric is not None):
                    if valid_scores[0] >= best_metric:
                        best_metric = valid_scores[0]
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                    if no_improve_epochs >= stop_no_improve_epochs and epoch >= 4:
                        logging.info("Tried the best, no improved and stop training")
                        break

            logging.info(log_str)

        # self.export_model(model_dir, only_pb=True)  # save the model

    def train_loop(self, train_loader):
        train_loss = 0.0
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
        gradients = [(tf.clip_by_value(grad, -5.0, 5.0)) for grad in gradients]
        _ = self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.global_step)
        else:
            lr = self.learning_rate
        self.optimizer.lr.assign(lr)
        self.global_step.assign_add(1)
        # logging.info('Step: {}, Loss: {}'.format(self.global_step.numpy(), loss))
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
        logging.info("Protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            logging.info("Model weights successfully saved in {}.ckpt".format(model_dir))


class KerasTrainer(object):
    """Keras trainer from tf.keras"""

    def __init__(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        loss_fn: Union[Callable] = tf.keras.losses.MeanSquaredError(),
        optimizer: tf.keras.optimizers = tf.keras.optimizers.legacy.Adam(0.003),
        lr_scheduler: Optional[tf.keras.optimizers.Optimizer] = None,
        strategy: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        run_eagerly: bool = True,
        **kwargs: Dict
    ) -> None:
        """
        model: tf.keras.Model instance
        loss: loss function
        optimizer: tf.keras.Optimizer instance
        run_eagerly: it depends on which one is much faster
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.strategy = strategy
        self.run_eagerly = run_eagerly

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(
        self,
        train_dataset,
        valid_dataset=None,
        epochs: int = 20,
        batch_size: int = 64,
        steps_per_epoch: Optional[int] = None,
        callback_metrics: Optional[List] = None,
        early_stopping: Optional[bool] = None,
        checkpoint=None,
        verbose: int = 2,
        **kwargs: Dict
    ):
        """
        train_dataset: tf.data.Dataset instance, or [x_train, y_train]
        valid_dataset: None or tf.data.Dataset instance, or [x_valid, y_valid]
        transform2label: transform function from logit to label
        """
        callbacks = []
        if early_stopping is not None:
            callbacks.append(early_stopping)
        if checkpoint is not None:
            callbacks.append(checkpoint)
        if "callbacks" in kwargs:
            callbacks += kwargs.get("callbacks")
            logging.info("callback", callbacks)

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
                if isinstance(x, dict):
                    inputs = {key: Input(item.shape[1:]) for key, item in x.items()}
                else:
                    inputs = Input(x.shape[1:])
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

            self.model = self.model.build_model(inputs=inputs)

        # print(self.model.summary())
        self.model.compile(
            loss=self.loss_fn, optimizer=self.optimizer, metrics=callback_metrics, run_eagerly=self.run_eagerly
        )
        if isinstance(train_dataset, (list, tuple)):
            x_train, y_train = train_dataset

            self.history = self.model.fit(
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
            self.history = self.model.fit(
                train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
            )
        return self.history

    def predict(self, x_test, batch_size: int = 1):
        y_test_pred = self.model(x_test)
        return y_test_pred

    def get_model(self):
        return self.model

    def save_model(self, model_dir, only_pb: bool = True, checkpoint_dir: Optional[str] = None):
        # save the model, checkpoint_dir if you use Checkpoint callback to save your best weights
        if checkpoint_dir is not None:
            logging.info("checkpoint Loaded", checkpoint_dir)
            self.model.load_weights(checkpoint_dir)
        else:
            logging.info("No checkpoint Loaded")

        self.model.save(model_dir)
        logging.info("protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            logging.info("model weights successfully saved in {}.ckpt".format(model_dir))
        return

    def plot(self, history, true, pred):
        import matplotlib.pyplot as plt

        train_length = history.shape[1]
        pred_length = true.shape[1]
        example = np.random.choice(range(history.shape[0]))

        plt.plot(range(train_length), history[example, :, 0], label="history")
        plt.plot(range(train_length, train_length + pred_length), true[example, :, 0], label="true")
        plt.plot(range(train_length, train_length + pred_length), pred[example, :, 0], label="pred")
        plt.legend()


class Seq2seqKerasTrainer(KerasTrainer):
    """As the transformers forum mentioned: https://discuss.huggingface.co/t/trainer-vs-seq2seqtrainer/3145/2
    Seq2SeqTrainer is mostly about predict_with_generate."""

    def __init__(self, *args, **kwargs):
        super(Seq2seqKerasTrainer, self).__init__(*args, **kwargs)
