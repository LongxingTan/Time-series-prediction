#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,  ModelCheckpoint, TensorBoard
from .models import Seq2seq, WaveNet, Transformer, Unet, NBeatsNet
from .loss import Loss
from .optimizer import Optimizer


class Model(object):
    """
    Trainer class
    """
    def __init__(self, params, use_model, use_loss='mse', use_optimizer='adam', custom_model_params={}):
        """

        :param params:
        :param use_model:
        :param use_loss:
        :param use_optimizer:
        :param custom_model_params:
        """
        self.params = params
        self.use_model = use_model
        self.use_loss = use_loss
        self.use_optimizer = use_optimizer
        self.custom_model_params = custom_model_params
        self.input_seq_length = params['input_seq_length']
        self.output_seq_length = params['output_seq_length']

    def build_model(self, training):
        """
        build the model instance, with loss and optimizer
        Args:
            training: if it's training, bool
        """
        if self.use_model == 'seq2seq':
            Model = Seq2seq(self.custom_model_params)
            inputs = Input([self.input_seq_length, 1])
            outputs = Model(inputs, training=training, predict_seq_length=self.output_seq_length)
        elif self.use_model == 'wavenet':
            Model = WaveNet(self.custom_model_params)
            inputs = Input([self.input_seq_length, 1])
            outputs = Model(inputs, training=training, predict_seq_length=self.output_seq_length)
        elif self.use_model == 'transformer':
            Model = Transformer(self.custom_model_params)
            inputs = (Input([self.input_seq_length, 1]), Input([self.output_seq_length, 1]))
            outputs = Model(inputs, training=training, predict_seq_length=self.output_seq_length)
        elif self.use_model == 'unet':
            Model = Unet(self.custom_model_params)
            inputs = Input([self.input_seq_length, 1])
            outputs = Model(inputs, training=training, predict_seq_length=self.output_seq_length)
        elif self.use_model == 'nbeats':
            Model = NBeatsNet(self.custom_model_params)
            inputs = Input([self.input_seq_length])
            outputs = Model(inputs, training=True, predict_seq_length=self.output_seq_length)
        else:
            raise ValueError("unsupported use_model of {} yet".format(self.use_model))

        self.model = tf.keras.Model(inputs, outputs, name=self.use_model)
        self.loss_fn = Loss(self.use_loss)()
        self.optimizer_fn = Optimizer(self.use_optimizer)(learning_rate=self.params['learning_rate'])

    def train(self, dataset, valid_dataset=None, n_epochs=10, mode='eager'):
        """
        train function
        :param dataset: train data loader to feed model
        :param valid_dataset: valid data loaded to feed model for evaluation
        :param n_epochs: number of epochs
        :param mode: training mode, `eager` or `fit`
        :return:
        """
        print("-" * 35)
        print("Start to train {}, in {} mode".format(self.use_model, mode))
        print("-" * 35)
        self.build_model(training=True)

        if mode == 'eager':
            self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            self.log_writer = tf.summary.create_file_writer(self.params['log_dir'])

            for epoch in range(1, n_epochs + 1):
                print("-> EPOCH {}".format(epoch))
                epoch_loss_train, epoch_loss_valid = [], []
                for step, (x, y) in enumerate(dataset.take(-1)):
                    loss = self.train_step(x, y)
                    print("=> STEP %4d  lr: %.6f  loss: %4.2f" % (self.global_steps, self.optimizer_fn.lr.numpy(), loss))

                    with self.log_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.global_steps)
                    epoch_loss_train.append(loss)

                with self.log_writer.as_default():
                    tf.summary.scalar('epoch_loss_train', tf.reduce_mean(epoch_loss_train), step=epoch)

                if valid_dataset is not None:
                    self.eval(valid_dataset)

            self.export_model()

        elif mode == 'fit':
            self.model.compile(loss=self.loss_fn, optimizer=self.optimizer_fn)
            callbacks = [TensorBoard(log_dir=self.params['log_dir']),
                         ModelCheckpoint(self.params['model_dir'], verbose=1, save_weights_only=True)
                         ]
            self.model.fit(dataset, epochs=n_epochs, callbacks=callbacks)
        else:
            raise ValueError("Unsupported train mode: {}, choose 'eager' or 'fit'".format(mode))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            try:
                y_pred = self.model(tf.cast(x, tf.float32), training=True)
            except:
                y_pred = self.model([tf.cast(x, tf.float32), tf.cast(y, tf.float32)], training=True)
            loss = self.loss_fn(y, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -10.0, 10.0)) for grad in gradients]
        self.optimizer_fn.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.global_steps.assign_add(1)
        return loss

    def eval(self, valid_dataset, eval_metrics=()):
        # self.build_model(training=False)
        # self.model.load_weights(self.params['model_dir'])

        for step, (x, y) in enumerate(valid_dataset.take(-1)):
            metrics = self.valid_step(x, y)
            print("=> STEP %4d Metrics: %4.2f" % (step, metrics))

        if eval_metrics:
            self.export_model()

    def valid_step(self, x, y):
        """
        evaluation step function
        :param x:
        :param y:
        """
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        try:
            y_pred = self.model(x, training=False)
        except:
            y_pred = self.model((x, tf.ones([tf.shape(x)[0], self.output_seq_length, 1], tf.float32)))
        metrics = self.loss_fn(y, y_pred).numpy()
        return metrics

    def predict(self, x_test, model_dir, use_model='pb'):
        """
        predict function, it doesn't use self.model here, but saved checkpoint or pb
        :param x_test:
        :param model_dir:
        :param use_model:
        :return:
        """
        if use_model == 'pb':
            print('Load saved pb model ...')
            model = tf.saved_model.load(model_dir)
        else:
            print('Load checkpoint model ...')
            model = self.model.load_weights(model_dir)

        y_pred = model(x_test)  # True, None, if x_test has more elements, it should use tuple
        return y_pred

    def export_model(self, only_pb=True):
        """
        save the model to .pb file for prediction
        """
        if not only_pb:
            self.model.save_weights(self.params['model_dir'])
            print("weights save in {}".format(self.params['model_dir']))
        tf.saved_model.save(self.model, self.params['saved_model_dir'])
        print("pb_model save in {}".format(self.params['saved_model_dir']))
