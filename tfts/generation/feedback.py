"""Selection of decoder inputs, independent of forecast sampling."""

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class FeedbackPolicy:
    """Choose teacher blocks with a per-example Bernoulli decision.

    Teachers are selected only *after* the corresponding prediction is made.
    Missing teacher elements fall back to the model value. A tensor probability
    can be supplied by an optimizer-step schedule without retracing the decoder.
    """

    teacher_probability: object = 0.0
    detach_predictions: bool = True

    def select(self, prediction, teacher=None, observed_mask=None, *, offset=0, seed=None):
        probability = tf.cast(self.teacher_probability, tf.float32)
        tf.debugging.assert_greater_equal(probability, 0.0)
        tf.debugging.assert_less_equal(probability, 1.0)
        value = tf.stop_gradient(prediction) if self.detach_predictions else prediction
        if teacher is None:
            tf.debugging.assert_equal(probability, 0.0, message="teacher_probability requires teacher targets")
            return value
        width = tf.shape(value)[1]
        target = tf.cast(teacher[:, offset : offset + width, ...], value.dtype)
        tf.debugging.assert_equal(tf.shape(target), tf.shape(value), message="teacher block shape mismatch")
        shape = tf.concat([tf.shape(value)[:1], tf.ones([tf.rank(value) - 1], tf.int32)], axis=0)
        uniform = (
            tf.random.uniform(shape)
            if seed is None
            else tf.random.stateless_uniform(shape, tf.random.experimental.stateless_fold_in(seed, 1))
        )
        use_teacher = uniform < probability
        if observed_mask is not None:
            use_teacher = use_teacher & tf.cast(observed_mask[:, offset : offset + width, ...], tf.bool)
        return tf.where(use_teacher, target, value)
