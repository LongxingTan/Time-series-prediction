#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf


class Optimizer(object):
    def __init__(self, use_optimizer):
        self.use_optimizer = use_optimizer

    def __call__(self, learning_rate):
        if self.use_optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr=learning_rate)
        elif self.use_optimizer == 'sgd':
            return tf.keras.optimizers.SGD(lr=learning_rate)
        else:
            raise ValueError("Not supported use_optimizer: {}".format(self.use_optimizer))


class LrScheduler(object):
    def __init__(self, total_steps, params, scheduler_method='cosine'):
        if scheduler_method == 'step':
            self.scheduler = Step(total_steps, params)
        elif scheduler_method == 'cosine':
            self.scheduler = Cosine(total_steps, params)
        self.step_count = 0
        self.total_steps = total_steps

    def step(self):
        self.step_count += 1
        lr = self.scheduler(self.step_count)
        return lr


class Step(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, params):
        # create the step learning rate with linear warmup
        super(Step, self).__init__()
        self.total_steps = total_steps
        self.params = params

    def __call__(self, global_step):
        warmup_lr = self.params['warmup_learning_rate']
        warmup_steps = self.params['warmup_steps']
        init_lr = self.params['init_learning_rate']
        lr_levels = self.params['learning_rate_levels']
        lr_steps = self.params['learning_rate_steps']
        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

        for next_learning_rate, start_step in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)

        return learning_rate


class Cosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, params):
        # create the cosine learning rate with linear warmup
        super(Cosine, self).__init__()
        self.total_steps = total_steps
        self.params = params

    def __call__(self, global_step):
        init_lr = self.params['init_learning_rate']
        warmup_lr = self.params['warmup_learning_rate'] if 'warmup_learning_rate' in self.params else 0.0
        warmup_steps = self.params['warmup_steps']
        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        cosine_learning_rate = init_lr * (
                    tf.cos(np.pi * (global_step - warmup_steps) / (self.total_steps - warmup_steps)) + 1.0) / 2.0
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, cosine_learning_rate)
        return learning_rate
