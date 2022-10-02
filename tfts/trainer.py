import numpy as np
import tensorflow as tf

__all__ = ["Trainer", "KerasTrainer"]


class Trainer(object):
    def __init__(self, model, loss_fn, optimizer, lr_scheduler=None, metrics=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics

    def train(
        self,
        train_loader,
        valid_loader,
        n_epochs=10,
        batch_size=8,
        learning_rate=3e-4,
        verbose=1,
        eval_metric=(),
        model_dir=None,
        use_ema=False,
        stop_no_improve_epochs=None,
        transform=None,
    ):
        """train function
        :param train_loader: tf.data.Dataset instance
        :param valid_loader: valid_dataset: None or tf.data.Dataset instance
        :param n_epochs:
        :param learning_rate:
        :param verbose:
        :param eval_metric:
        :param model_dir:
        :param stop_no_improve_epochs: if None, no early stop; otherwise, training will stop after no_improve_epochs
        based on the 1st eval_metric score
        """
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric
        self.use_ema = use_ema
        self.transform = transform
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

        if use_ema:
            self.ema = tf.train.ExponentialMovingAverage(0.9).apply(self.model.trainable_variables)

        if model_dir is None:
            model_dir = "../weights"

        if stop_no_improve_epochs is not None:
            no_improve_epochs = 0
            best_metric = -np.inf

        for epoch in range(n_epochs):
            train_loss, train_scores = self.train_loop(train_loader)
            if valid_loader is not None:
                valid_loss, valid_scores = self.valid_loop(valid_loader)
            else:
                valid_loss = 999
                valid_scores = [999]

            log_str = "Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}".format(epoch + 1, train_loss, valid_loss)
            log_str + ",".join([" Valid Metrics{}: {:.4f}".format(i, me) for i, me in enumerate(valid_scores)])
            print(log_str)

            if stop_no_improve_epochs is not None:
                if valid_scores[0] >= best_metric:
                    best_metric = valid_scores[0]
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                if no_improve_epochs >= stop_no_improve_epochs and epoch >= 4:
                    print("I have tried my best, no improved and stop training!")
                    break

        self.export_model(model_dir, only_pb=True)  # save the model

    def train_loop(self, train_loader):
        train_loss = 0.0
        y_trues, y_preds = [], []

        for step, (x_train, y_train) in enumerate(train_loader):
            y_pred, step_loss = self.train_step(x_train, y_train)
            train_loss += step_loss
            y_preds.append(y_pred)
            y_trues.append(y_train)

        scores = []
        if self.eval_metric:
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
        # print('Step: {}, Loss: {}'.format(self.global_step.numpy(), loss))
        return y_pred, loss

    def valid_loop(self, valid_loader):
        valid_loss = 0.0
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
        print("protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            print("model weights successfully saved in {}.ckpt".format(model_dir))


class KerasTrainer(object):
    def __init__(
        self,
        build_model,
        loss_fn=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(0.003),
        lr_scheduler=None,
        strategy=None,
    ):
        """
        model: a tf.keras.Model instance
        loss: a loss function
        optimizer: tf.keras.Optimizer instance
        """
        self.build_model = build_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.strategy = strategy

    def train(
        self,
        train_dataset,
        valid_dataset=None,
        n_epochs=10,
        batch_size=32,
        steps_per_epoch=None,
        callback_eval_metrics=None,
        transform=None,
        early_stopping=None,
        checkpoint=None,
        verbose=2,
        **kwargs
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
            print("callback", callbacks)

        # if self.strategy is None:
        #     self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        # else:
        #     train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        #     if valid_dataset is not None:
        #         valid_dataset = self.strategy.experimental_distribute_dataset(valid_dataset)

        # with self.strategy.scope():
        self.model = self.build_model()
        print(self.model.summary())
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=callback_eval_metrics, run_eagerly=True)
        if isinstance(train_dataset, (list, tuple)):
            x_train, y_train = train_dataset
            x_valid, y_valid = valid_dataset
            self.history = self.model.fit(
                x_train,
                y_train,
                validation_data=(x_valid, y_valid),
                steps_per_epoch=steps_per_epoch,
                epochs=n_epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
            )
        else:
            self.history = self.model.fit(
                train_dataset,
                validation_data=valid_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=n_epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
            )
        return self.history

    def predict(self, x_test, method=None, batch_size=1):
        y_test_pred = self.model.predict(x_test, batch_size=batch_size)
        return y_test_pred

    def get_model(self):
        return self.model

    def save_model(self, model_dir, only_pb=True, checkpoint_dir=None):
        # save the model
        if checkpoint_dir is not None:
            print("check", checkpoint_dir)
            self.model.load_weights(checkpoint_dir)
        else:
            print("nocheck")

        self.model.save(model_dir)
        print("protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            print("model weights successfully saved in {}.ckpt".format(model_dir))
        return
