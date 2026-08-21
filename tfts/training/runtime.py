"""Runtime setup for training.

This module keeps accelerator and precision choices in one place so trainers,
pipelines, and future custom loops can share the same behavior.
"""

import logging
from typing import Optional

import tensorflow as tf

from ..training_args import TrainingArguments

logger = logging.getLogger(__name__)


def _create_mirrored_strategy() -> tf.distribute.Strategy:
    """Create MirroredStrategy when this TensorFlow build supports it."""
    mirrored_strategy = getattr(tf.distribute, "MirroredStrategy", None)
    if mirrored_strategy is None:
        logger.warning("MirroredStrategy is not available in this TensorFlow build; using default strategy.")
        return tf.distribute.get_strategy()
    return mirrored_strategy()


def create_distribution_strategy(args: Optional[TrainingArguments] = None) -> tf.distribute.Strategy:
    """Create a TensorFlow distribution strategy from training arguments.

    Args:
        args: Training arguments. If omitted, uses automatic local device detection.

    Returns:
        A TensorFlow distribution strategy.
    """
    strategy_name = args.strategy if args is not None else "auto"

    if strategy_name == "default":
        return tf.distribute.get_strategy()

    if strategy_name == "multi_worker":
        logger.info("Using MultiWorkerMirroredStrategy")
        return tf.distribute.MultiWorkerMirroredStrategy()

    gpus = tf.config.list_physical_devices("GPU")

    if strategy_name == "mirrored":
        logger.info("Using MirroredStrategy")
        return _create_mirrored_strategy()

    if strategy_name == "one_device":
        device = "/gpu:0" if gpus else "/cpu:0"
        logger.info("Using OneDeviceStrategy on %s", device)
        return tf.distribute.OneDeviceStrategy(device=device)

    if len(gpus) > 1:
        logger.info("Using MirroredStrategy with %s GPUs", len(gpus))
        return _create_mirrored_strategy()
    if len(gpus) == 1:
        # Use the default strategy on a single GPU. The default TF strategy already runs ops
        # on the available GPU, and TensorFlow's Model.fit raises "Mixing different
        # tf.distribute.Strategy objects" when fed numpy inputs / odd-sized validation batches
        # inside a OneDeviceStrategy scope. So on one device we do not need (and must not use)
        # a promoting device strategy.
        logger.info("Using default TensorFlow strategy (single GPU)")
        return tf.distribute.get_strategy()

    logger.info("Using default TensorFlow strategy")
    return tf.distribute.get_strategy()


def configure_precision(args: TrainingArguments) -> tf.keras.mixed_precision.Policy:
    """Apply the configured Keras mixed precision policy.

    Args:
        args: Training arguments containing the precision setting.

    Returns:
        The active mixed precision policy.
    """
    policy = tf.keras.mixed_precision.Policy(args.precision)
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Using precision policy: %s", policy.name)
    return policy
