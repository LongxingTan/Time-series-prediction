import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModel, AutoModelForTimeSeriesClassification
from tfts.trainer import BaseTrainer, EagerTrainer, KerasTrainer, Seq2seqKerasTrainer, Trainer, set_seed
from tfts.training.runtime import configure_precision, create_distribution_strategy
from tfts.training_args import TrainingArguments


def _gpu_count() -> int:
    """Return the number of physical GPUs visible to TensorFlow."""
    return len(tf.config.list_physical_devices("GPU"))


N_GPUS = _gpu_count()

# Full-distribution training only makes sense when a real multi-GPU setup exists.
# GitHub Actions runners have no GPU, so these tests exercise the single-device path
# there and run the true multi-GPU path on local machines with >= 2 GPUs.
needs_multigpu = unittest.skipUnless(N_GPUS >= 2, "requires >= 2 GPUs")

# Real fp16 mixed-precision training needs an accelerator; TF will often error out
# for fp16 ops on a CPU runner.
needs_gpu = unittest.skipUnless(N_GPUS >= 1, "requires a GPU")


def _tfts_trainer(model, cls=KerasTrainer, **kwargs):
    """Build a trainer pinned to a single-device ("default") strategy.

    The functional tests below validate trainer *logic* (optimizers, metrics,
    callbacks, saving, ...), not distribution. They must therefore be deterministic
    regardless of how many GPUs happen to be visible on the host (GitHub runners have
    none; multi-GPU machines may have several). Real multi-GPU behavior is exercised
    separately by the ``needs_multigpu``-guarded tests.
    """
    kwargs.setdefault("args", TrainingArguments(output_dir="./weights", strategy="default"))
    return cls(model, **kwargs)


class SetSeedTest(unittest.TestCase):
    """Test the set_seed utility function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        set_seed(42)
        random_val1 = np.random.random()
        tf_random_val1 = tf.random.normal([1]).numpy()[0]

        set_seed(42)
        random_val2 = np.random.random()
        tf_random_val2 = tf.random.normal([1]).numpy()[0]

        self.assertEqual(random_val1, random_val2)
        self.assertEqual(tf_random_val1, tf_random_val2)

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        random_val1 = np.random.random()

        set_seed(123)
        random_val2 = np.random.random()

        self.assertNotEqual(random_val1, random_val2)


class BaseTrainerTest(unittest.TestCase):
    """Test BaseTrainer functionality."""

    def setUp(self):
        self.config = AutoConfig.for_model("rnn")
        self.model = AutoModel.from_config(self.config, predict_sequence_length=2)

    def test_initialization_with_defaults(self):
        """Test BaseTrainer initialization with default arguments."""
        trainer = BaseTrainer(self.model)
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.args)
        self.assertIsNotNone(trainer.strategy)

    def test_initialization_with_custom_args(self):
        """Test BaseTrainer initialization with custom training arguments."""
        custom_args = TrainingArguments(
            output_dir="./custom_output", learning_rate=0.001, per_device_train_batch_size=16
        )
        trainer = BaseTrainer(self.model, args=custom_args)
        self.assertEqual(trainer.args.learning_rate, 0.001)
        self.assertEqual(trainer.args.per_device_train_batch_size, 16)

    def test_get_strategy_scope(self):
        """Test strategy scope context manager."""
        trainer = BaseTrainer(self.model)
        with trainer.get_strategy_scope():
            # Should not raise any errors
            pass

    def test_create_optimizer(self):
        """Test optimizer creation with default parameters."""
        trainer = BaseTrainer(self.model)
        optimizer = trainer._create_optimizer()
        self.assertIsInstance(optimizer, tf.keras.optimizers.Optimizer)

    def test_create_lr_scheduler_linear(self):
        """Test linear learning rate scheduler creation."""
        args = TrainingArguments(output_dir="./test", lr_scheduler_type="linear", max_steps=100)
        trainer = BaseTrainer(self.model, args=args)
        scheduler = trainer._create_lr_scheduler()
        self.assertIsInstance(scheduler, tf.keras.optimizers.schedules.LearningRateSchedule)

    def test_create_lr_scheduler_none(self):
        """Test that no scheduler is created when type is not specified."""
        args = TrainingArguments(output_dir="./test", lr_scheduler_type="none", max_steps=100)
        trainer = BaseTrainer(self.model, args=args)
        scheduler = trainer._create_lr_scheduler()
        self.assertIsNone(scheduler)

    def test_get_inputs_from_dataset(self):
        """Test input preparation from tf.data.Dataset."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.random((2, 2, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)

        trainer = BaseTrainer(self.model)
        inputs = trainer.get_inputs(dataset)
        # Check if it's a tensor OR a KerasTensor (which behaves differently in different TF versions)
        is_tensor = tf.is_tensor(inputs)
        is_keras_tensor = tf.keras.backend.is_keras_tensor(inputs)
        self.assertTrue(is_tensor or is_keras_tensor)

    def test_get_inputs_from_sequence(self):
        """Test input preparation from keras.utils.Sequence."""

        class DummySequence(tf.keras.utils.Sequence):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return np.random.random((1, 10, 1)), np.random.random((1, 2, 1))

        sequence = DummySequence()
        trainer = BaseTrainer(self.model)
        inputs = trainer.get_inputs(sequence)
        is_tensor = tf.is_tensor(inputs)
        is_keras_tensor = tf.keras.backend.is_keras_tensor(inputs)
        self.assertTrue(is_tensor or is_keras_tensor)

    def test_get_inputs_from_list(self):
        """Test input preparation from list/tuple."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.random((2, 2, 1))
        dataset = (x_train, y_train)

        trainer = BaseTrainer(self.model)
        inputs = trainer.get_inputs(dataset)
        is_tensor = tf.is_tensor(inputs)
        is_keras_tensor = tf.keras.backend.is_keras_tensor(inputs)
        self.assertTrue(is_tensor or is_keras_tensor)

    def test_get_inputs_dict(self):
        """Test input preparation from dictionary data."""
        x_dict = {"input1": np.random.random((2, 10, 1)), "input2": np.random.random((2, 5, 1))}
        y_train = np.random.random((2, 2, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x_dict, y_train)).batch(1)

        trainer = BaseTrainer(self.model)
        inputs = trainer.get_inputs(dataset)
        self.assertIsInstance(inputs, dict)

    def test_get_inputs_multiple_arrays(self):
        """Test input preparation from multiple input arrays."""
        x1 = np.random.random((2, 10, 1))
        x2 = np.random.random((2, 5, 1))
        y_train = np.random.random((2, 2, 1))
        dataset = tf.data.Dataset.from_tensor_slices(((x1, x2), y_train)).batch(1)

        trainer = BaseTrainer(self.model)
        inputs = trainer.get_inputs(dataset)
        self.assertIsInstance(inputs, list)

    def test_get_inputs_invalid_type(self):
        """Test that invalid dataset type raises ValueError."""
        trainer = BaseTrainer(self.model)
        with self.assertRaises(ValueError):
            trainer.get_inputs("invalid_type")

    def test_global_batch_size(self):
        """Test global batch size calculation."""
        args = TrainingArguments(output_dir="./test", per_device_train_batch_size=8)
        trainer = BaseTrainer(self.model, args=args)
        batch_size = trainer.global_batch_size
        self.assertGreater(batch_size, 0)

    def test_create_optimizer_uses_training_arguments(self):
        """Test optimizer creation respects TrainingArguments."""
        args = TrainingArguments(output_dir="./test", learning_rate=0.002, weight_decay=0.01, lr_scheduler_type="none")
        trainer = BaseTrainer(self.model, args=args)
        optimizer = trainer._create_optimizer()
        self.assertAlmostEqual(float(tf.keras.backend.get_value(optimizer.learning_rate)), 0.002)

    def test_save_model(self):
        """Test model saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BaseTrainer(self.model)
            trainer.model(tf.zeros([1, 10, 1]))
            trainer._save(tmpdir)
            # Check that config file exists
            config_path = os.path.join(tmpdir, "config.json")
            self.assertTrue(os.path.exists(config_path))
            weights_path = os.path.join(tmpdir, "tf_model.weights.h5")
            self.assertTrue(os.path.exists(weights_path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "task_config.json")))

    def test_save_unbuilt_model_raises_clear_error(self):
        """Test saving an unbuilt model fails before writing partial files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BaseTrainer(self.model)
            with self.assertRaisesRegex(ValueError, "cannot be saved before the model is built"):
                trainer._save(tmpdir)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "config.json")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "tf_model.weights.h5")))


class TrainingRuntimeTest(unittest.TestCase):
    """Test training runtime configuration helpers."""

    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_training_arguments_fp16_sets_precision(self):
        """Test fp16 compatibility flag maps to mixed_float16 policy."""
        args = TrainingArguments(output_dir="./test", fp16=True)
        self.assertEqual(args.precision, "mixed_float16")

    def test_training_arguments_bf16_sets_precision(self):
        """Test bf16 compatibility flag maps to mixed_bfloat16 policy."""
        args = TrainingArguments(output_dir="./test", bf16=True)
        self.assertEqual(args.precision, "mixed_bfloat16")

    def test_training_arguments_rejects_conflicting_precision_flags(self):
        """Test fp16 and bf16 cannot both be enabled."""
        with self.assertRaises(ValueError):
            TrainingArguments(output_dir="./test", fp16=True, bf16=True)

    def test_configure_precision(self):
        """Test precision policy is applied globally."""
        args = TrainingArguments(output_dir="./test", precision="mixed_float16")
        policy = configure_precision(args)
        self.assertEqual(policy.name, "mixed_float16")
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "mixed_float16")

    @patch("tfts.training.runtime.tf.distribute.MirroredStrategy", create=True)
    @patch("tensorflow.config.list_physical_devices")
    def test_create_distribution_strategy_auto_multi_gpu(self, mock_list_devices, mock_mirrored_strategy):
        """Test automatic strategy selection for multiple GPUs."""
        mock_list_devices.return_value = ["GPU:0", "GPU:1"]
        strategy = create_distribution_strategy(TrainingArguments(output_dir="./test"))
        mock_mirrored_strategy.assert_called_once_with()
        self.assertEqual(strategy, mock_mirrored_strategy.return_value)

    @patch("tensorflow.config.list_physical_devices")
    def test_create_distribution_strategy_auto_cpu(self, mock_list_devices):
        """Test automatic strategy selection on CPU."""
        mock_list_devices.return_value = []
        strategy = create_distribution_strategy(TrainingArguments(output_dir="./test"))
        self.assertIsInstance(strategy, tf.distribute.Strategy)


class EagerTrainerTest(unittest.TestCase):
    """Tests for EagerTrainer (legacy custom training loop)."""

    def setUp(self):
        self.fit_config = {
            "epochs": 2,
            "stop_no_improve_epochs": 1,
            "eval_metric": lambda x, y: np.mean(np.abs(x.numpy() - y.numpy())),
            "model_dir": "./weights",
        }

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        self.train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        self.valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        for x, y in self.train_loader:
            self.assertEqual(x.shape, (1, 10, 1))
            break

    def tearDown(self):
        if os.path.exists("./weights"):
            shutil.rmtree("./weights", ignore_errors=True)

    def test_trainer_basic(self):
        # 1gpu, no dist
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(
            model,
        )
        trainer.train(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            optimizer=tf.keras.optimizers.Adam(0.003),
            **self.fit_config,
        )
        trainer.predict(self.valid_loader)
        trainer.save_model(model_dir="./weights", only_pb=True)

    def test_trainer_fit_alias(self):
        """Test that fit() is an alias for train()."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        # fit should work the same as train
        trainer.fit(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            optimizer=tf.keras.optimizers.Adam(0.003),
            epochs=1,
        )

    def test_trainer_without_validation(self):
        """Test training without validation data."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        trainer.train(
            train_loader=self.train_loader, valid_loader=None, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1
        )

    def test_trainer_with_lr_scheduler(self):
        """Test trainer with learning rate scheduler."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.003, decay_steps=10, decay_rate=0.9
        )

        trainer.train(
            train_loader=self.train_loader, valid_loader=self.valid_loader, lr_scheduler=lr_schedule, epochs=1
        )

    def test_trainer_with_ema(self):
        """Test trainer with exponential moving average."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, use_ema=True, epochs=1)

    def test_trainer_with_multiple_metrics(self):
        """Test trainer with multiple evaluation metrics."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        metrics = [
            lambda x, y: np.mean(np.abs(x.numpy() - y.numpy())),
            lambda x, y: np.mean(np.square(x.numpy() - y.numpy())),
        ]

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, eval_metric=metrics, epochs=1)

    def test_trainer_early_stopping(self):
        """Test early stopping functionality."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        trainer.train(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            stop_no_improve_epochs=1,
            eval_metric=lambda x, y: np.mean(np.abs(x.numpy() - y.numpy())),
            epochs=10,  # Should stop early
        )

    def test_trainer_gradient_clipping(self):
        """Test gradient clipping with custom max_grad_norm."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, max_grad_norm=1.0, epochs=1)

    def test_trainer_custom_loss(self):
        """Test trainer with custom loss function."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        custom_loss = tf.keras.losses.MeanAbsoluteError()

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, loss_fn=custom_loss, epochs=1)

    @needs_multigpu
    def test_trainer_2gpu(self):
        strategy = tf.distribute.MirroredStrategy()
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model, strategy=strategy)
        self.assertGreater(strategy.num_replicas_in_sync, 1)
        trainer.train(self.train_loader, self.valid_loader, **self.fit_config)

    def test_trainer_multi_gpu_strategy_replica_count(self):
        """Verify MirroredStrategy replicates across devices when GPUs are present.

        Runs on both CI (single CPU device) and local multi-GPU boxes. On a real
        multi-GPU machine we assert replication is actually active.
        """
        strategy = tf.distribute.MirroredStrategy()
        self.assertGreaterEqual(strategy.num_replicas_in_sync, 1)
        if N_GPUS >= 2:
            self.assertGreater(strategy.num_replicas_in_sync, 1)

    def test_trainer_gradient_accumulation(self):
        """Test gradient accumulation matches a normal update in terms of step count."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model)

        # 2 micro-batches, accumulate over 2 steps -> a single optimizer step.
        trainer.train(
            train_loader=self.train_loader,
            valid_loader=None,
            optimizer=tf.keras.optimizers.Adam(0.003),
            gradient_accumulation_steps=2,
            epochs=1,
        )
        # 2 micro-batches / 2 accumulation steps = 1 actual update.
        self.assertEqual(int(trainer.global_step.numpy()), 1)

    @needs_gpu
    def test_trainer_mixed_precision_fp16(self):
        """Real fp16 mixed-precision training (requires a GPU)."""
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            config = AutoConfig.for_model("rnn")
            model = AutoModel.from_config(config, predict_sequence_length=2)
            trainer = EagerTrainer(model)
            trainer.train(
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                optimizer=tf.keras.optimizers.Adam(0.003),
                epochs=1,
            )
            self.assertEqual(tf.keras.mixed_precision.global_policy().name, "mixed_float16")
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    @unittest.skipIf(
        os.name == "nt" and tf.__version__.startswith("2.21."),
        "TensorFlow 2.21 bfloat16 CPU kernels can crash with SIGILL on Windows",
    )
    def test_trainer_mixed_precision_bf16_cpu_and_gpu(self):
        """bf16 mixed-precision training runs on both CPU and GPU (safe for CI)."""
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        try:
            config = AutoConfig.for_model("rnn")
            model = AutoModel.from_config(config, predict_sequence_length=2)
            trainer = EagerTrainer(model)
            trainer.train(
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                optimizer=tf.keras.optimizers.Adam(0.003),
                epochs=1,
            )
            self.assertEqual(tf.keras.mixed_precision.global_policy().name, "mixed_bfloat16")
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")

    def test_trainer_kwargs(self):
        """Test that custom kwargs are set as attributes."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = EagerTrainer(model, custom_param="test_value", another_param=42)

        self.assertEqual(trainer.custom_param, "test_value")
        self.assertEqual(trainer.another_param, 42)


class KerasTrainerTest(unittest.TestCase):
    def setUp(self):
        self.fit_config = {
            "epochs": 1,
            "batch_size": 1,
        }

    def tearDown(self):
        if os.path.exists("./weights"):
            shutil.rmtree("./weights", ignore_errors=True)

    def test_trainer_basic_array(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        trainer = _tfts_trainer(model)
        trainer.train(
            train_dataset=(x_train, y_train),
            valid_dataset=(x_valid, y_valid),
            optimizer=tf.keras.optimizers.Adam(0.003),
            **self.fit_config,
        )
        y_valid_pred = trainer.predict(x_valid)
        self.assertEqual(y_valid_pred.shape, (1, 2, 1))

    def test_trainer_basic_tfdata(self):
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        x_valid = np.random.random((1, 10, 1))
        y_valid = np.random.randint(0, 2, (1, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
        valid_loader = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(1)

        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)
        trainer.train(train_loader, valid_loader, optimizer=tf.keras.optimizers.Adam(0.003), **self.fit_config)
        trainer.save_model("./weights")

    def test_trainer_fit_alias(self):
        """Test that fit() is an alias for train()."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        history = trainer.fit(train_dataset=(x_train, y_train), epochs=1, batch_size=1)
        self.assertIsNotNone(history)

    def test_trainer_with_string_optimizer(self):
        """Test training with optimizer specified as string."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(train_dataset=(x_train, y_train), optimizer="adam", epochs=1, batch_size=1)

    def test_trainer_with_dict_optimizer(self):
        """Test training with optimizer specified as dict."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(
            train_dataset=(x_train, y_train),
            optimizer={"class_name": "Adam", "config": {"learning_rate": 0.001}},
            epochs=1,
            batch_size=1,
        )

    def test_trainer_with_string_loss(self):
        """Test training with loss function specified as string."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(train_dataset=(x_train, y_train), loss_fn="mae", epochs=1, batch_size=1)

    def test_trainer_with_metrics(self):
        """Test training with metrics."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(train_dataset=(x_train, y_train), metrics=["mae", "mse"], epochs=1, batch_size=1)

    def test_trainer_with_callbacks(self):
        """Test training with custom callbacks."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)

        trainer.train(train_dataset=(x_train, y_train), callbacks=[early_stopping], epochs=5, batch_size=1)

    def test_trainer_with_steps_per_epoch(self):
        """Test training with custom steps_per_epoch."""
        x_train = np.random.random((10, 10, 1))
        y_train = np.random.randint(0, 2, (10, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(2)

        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(train_dataset=train_loader, steps_per_epoch=2, epochs=1)

    def test_trainer_run_eagerly(self):
        """Test training with run_eagerly=True."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        trainer.train(train_dataset=(x_train, y_train), run_eagerly=True, epochs=1, batch_size=1)

    def test_trainer_verbose_levels(self):
        """Test training with different verbose levels."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        for verbose in [0, 1, 2]:
            trainer = _tfts_trainer(model)
            trainer.train(train_dataset=(x_train, y_train), verbose=verbose, epochs=1, batch_size=1)

    def test_get_model(self):
        """Test get_model() returns the correct model."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)

        retrieved_model = trainer.get_model()
        self.assertIsInstance(retrieved_model, tf.keras.Model)

    def test_evaluate_predict_and_default_task_helpers(self):
        model = tf.keras.Sequential([tf.keras.Input(shape=(4, 1)), tf.keras.layers.Dense(1)])
        trainer = _tfts_trainer(model)
        x = np.random.random((2, 4, 1)).astype(np.float32)
        y = np.random.random((2, 4, 1)).astype(np.float32)

        list_metrics = trainer.evaluate((x, y), metrics=["mae"])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
        dataset_metrics = trainer.evaluate(dataset, metrics=["mse"])
        np.testing.assert_equal(set(list_metrics), {"mae"})
        np.testing.assert_equal(set(dataset_metrics), {"mse"})
        self.assertEqual(trainer.predict(dataset).shape, x.shape)
        with self.assertRaises(TypeError):
            trainer.evaluate("invalid")

        self.assertIsInstance(trainer._default_loss(), tf.keras.losses.MeanSquaredError)
        self.assertEqual(trainer._default_metrics(), ["mae"])

        classifier = AutoModelForTimeSeriesClassification.from_config(AutoConfig.for_model("bert"), num_labels=3)
        classification_trainer = _tfts_trainer(classifier)
        self.assertIsInstance(
            classification_trainer._default_loss(),
            tf.keras.losses.SparseCategoricalCrossentropy,
        )
        self.assertEqual([metric.name for metric in classification_trainer._default_metrics()], ["accuracy"])

    def test_build_callbacks_covers_optional_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = KerasTrainer._build_callbacks(
                early_stopping_patience=1,
                checkpoint_dir=tmpdir,
                reduce_lr_patience=2,
            )
        self.assertEqual(len(callbacks), 3)
        self.assertIsInstance(callbacks[0], tf.keras.callbacks.EarlyStopping)
        self.assertIsInstance(callbacks[1], tf.keras.callbacks.ModelCheckpoint)
        self.assertIsInstance(callbacks[2], tf.keras.callbacks.ReduceLROnPlateau)

    def test_plot(self):
        """Test plot functionality."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model)

        history = np.random.random((5, 10, 1))
        true = np.random.random((5, 5, 1))
        pred = np.random.random((5, 5, 1))

        # Just test that plot doesn't raise an error
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        trainer.plot(history, true, pred)

    def test_trainer_with_keras_model(self):
        """Test training with a pre-built Keras model."""
        keras_model = tf.keras.Sequential([tf.keras.layers.LSTM(32, input_shape=(10, 1)), tf.keras.layers.Dense(2)])

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.random((2, 2))

        trainer = _tfts_trainer(keras_model)
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)

    def test_trainer_kwargs(self):
        """Test that custom kwargs are set as attributes."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model, custom_attr="test", number_attr=123)

        self.assertEqual(trainer.custom_attr, "test")
        self.assertEqual(trainer.number_attr, 123)

    def test_save_model_distributed(self):
        """Test that non-chief workers don't save in distributed training."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        # Mock a distributed strategy with non-chief task
        mock_resolver = Mock()
        mock_resolver.task_type = "worker"

        mock_strategy = MagicMock()
        mock_strategy.cluster_resolver = mock_resolver
        mock_strategy.num_replicas_in_sync = 1

        trainer = KerasTrainer(model, strategy=mock_strategy)

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)

        # Non-chief should return without saving
        trainer.save_model("./weights")


class Seq2seqKerasTrainerTest(unittest.TestCase):
    """Test Seq2seqKerasTrainer."""

    def test_inheritance(self):
        """Test that Seq2seqKerasTrainer inherits from KerasTrainer."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = _tfts_trainer(model, cls=Seq2seqKerasTrainer)

        self.assertIsInstance(trainer, KerasTrainer)

    def test_basic_training(self):
        """Test basic training with Seq2seqKerasTrainer."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        trainer = _tfts_trainer(model, cls=Seq2seqKerasTrainer)
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)


if __name__ == "__main__":
    unittest.main()
