"""Runtime setup for training.

This module keeps accelerator and precision choices in one place so trainers,
pipelines, and future custom loops can share the same behavior.
"""

import logging
import platform
from typing import Any, Optional

import tensorflow as tf

from ..training_args import TrainingArguments

logger = logging.getLogger(__name__)


def create_adamw(learning_rate: Any, **kwargs: Any) -> tf.keras.optimizers.Optimizer:
    """Create an AdamW optimizer across legacy and current Keras runtimes.

    TensorFlow's legacy AdamW is considerably faster on Apple Silicon for the
    Keras 2 runtime used by TensorFlow 2.13/2.15. Keras 3 intentionally does
    not support that legacy namespace, so it falls back to the current AdamW.
    """
    optimizer_kwargs = dict(kwargs)
    optimizer_kwargs["learning_rate"] = learning_rate
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}

    if is_apple_silicon:
        try:
            legacy_adamw = tf.keras.optimizers.legacy.AdamW
        except (AttributeError, ImportError):
            legacy_adamw = None
        if legacy_adamw is not None:
            try:
                return legacy_adamw(**optimizer_kwargs)
            except (AttributeError, ImportError):
                pass
        try:
            legacy_adam = tf.keras.optimizers.legacy.Adam
        except (AttributeError, ImportError):
            legacy_adam = None
        if legacy_adam is not None:
            optimizer_kwargs.pop("weight_decay", None)
            try:
                return legacy_adam(**optimizer_kwargs)
            except (AttributeError, ImportError):
                pass

    try:
        return tf.keras.optimizers.AdamW(**optimizer_kwargs)
    except AttributeError:
        optimizer_kwargs.pop("weight_decay", None)
        return tf.keras.optimizers.Adam(**optimizer_kwargs)


_MIRRORED_STRATEGY_CACHE = {}


def _nccl_available() -> bool:
    """Return True if the NCCL library is loadable.

    ``MirroredStrategy`` uses NCCL for cross-GPU collective gradient reduction.
    When no NCCL is installed, building the strategy succeeds but every
    ``CollectiveReduceV2`` op aborts with ``NCCL: Unable to load NCCL library``.
    Probes the loader so callers can degrade to a single device instead.
    """
    import ctypes
    import ctypes.util

    for name in ("libnccl.so.2", "libnccl.so"):
        path = ctypes.util.find_library(name)
        if path is not None:
            try:
                ctypes.CDLL(path)
                return True
            except OSError:
                continue
    return False


def _create_mirrored_strategy() -> tf.distribute.Strategy:
    """Create (and reuse) a MirroredStrategy when this TensorFlow build supports it.

    The strategy is cached as a process-wide singleton. Building a *new*
    MirroredStrategy object each time a ``Trainer``/``KerasTrainer`` is created
    collides TensorFlow's per-(GPU, graph) collective executor across sequential
    ``fit`` calls with differently-shaped models (e.g. per-optuna-trial sizes),
    surfacing as ``CollectiveReduceV2`` shape mismatches. Reusing one strategy
    keeps the collective graph consistent.
    """
    mirrored_strategy = getattr(tf.distribute, "MirroredStrategy", None)
    if mirrored_strategy is None:
        logger.warning("MirroredStrategy is not available in this TensorFlow build; using default strategy.")
        return tf.distribute.get_strategy()
    cached = _MIRRORED_STRATEGY_CACHE.get("mirrored")
    # Include the factory identity so monkeypatching and TensorFlow runtime
    # replacement cannot accidentally return a strategy from another runtime.
    if cached is None or cached[0] is not mirrored_strategy:
        cached = (mirrored_strategy, mirrored_strategy())
        _MIRRORED_STRATEGY_CACHE["mirrored"] = cached
    return cached[1]


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
        if len(gpus) > 1 and not _nccl_available():
            # Degrade to the default strategy: MirroredStrategy across these
            # GPUs can't reduce gradients without NCCL and would abort on
            # CollectiveReduceV2. The default strategy keeps ops on the GPU
            # without cross-device collectives.
            logger.warning(
                "NCCL is unavailable; falling back to the default TensorFlow "
                "strategy instead of multi-GPU MirroredStrategy."
            )
            return tf.distribute.get_strategy()
        logger.info("Using MirroredStrategy")
        return _create_mirrored_strategy()

    if strategy_name == "one_device":
        device = "/gpu:0" if gpus else "/cpu:0"
        logger.info("Using OneDeviceStrategy on %s", device)
        return tf.distribute.OneDeviceStrategy(device=device)

    if len(gpus) > 1 and not _nccl_available():
        logger.warning("NCCL is unavailable; falling back to a single device instead of " "multi-GPU MirroredStrategy.")
        gpus = gpus[:1]
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
