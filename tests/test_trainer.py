import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModel
from tfts.trainer import BaseTrainer, KerasTrainer, Seq2seqKerasTrainer, Trainer, set_seed


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
        from tfts.training_args import TrainingArguments

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
        from tfts.training_args import TrainingArguments

        args = TrainingArguments(output_dir="./test", lr_scheduler_type="linear", max_steps=100)
        trainer = BaseTrainer(self.model, args=args)
        scheduler = trainer._create_lr_scheduler()
        self.assertIsInstance(scheduler, tf.keras.optimizers.schedules.LearningRateSchedule)

    def test_create_lr_scheduler_none(self):
        """Test that no scheduler is created when type is not specified."""
        from tfts.training_args import TrainingArguments

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
        from tfts.training_args import TrainingArguments

        args = TrainingArguments(output_dir="./test", per_device_train_batch_size=8)
        trainer = BaseTrainer(self.model, args=args)
        batch_size = trainer.global_batch_size
        self.assertGreater(batch_size, 0)

    def test_save_model(self):
        """Test model saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BaseTrainer(self.model)
            trainer._save(tmpdir)
            # Check that config file exists
            config_path = os.path.join(tmpdir, "config.json")
            self.assertTrue(os.path.exists(config_path))


class TrainerTest(unittest.TestCase):
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
        trainer = Trainer(
            model,
        )
        trainer.train(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            optimizer=tf.keras.optimizers.Adam(0.003),
            **self.fit_config
        )
        trainer.predict(self.valid_loader)
        trainer.save_model(model_dir="./weights", only_pb=True)

    def test_trainer_fit_alias(self):
        """Test that fit() is an alias for train()."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model)

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
        trainer = Trainer(model)

        trainer.train(
            train_loader=self.train_loader, valid_loader=None, optimizer=tf.keras.optimizers.Adam(0.003), epochs=1
        )

    def test_trainer_with_lr_scheduler(self):
        """Test trainer with learning rate scheduler."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model)

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
        trainer = Trainer(model)

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, use_ema=True, epochs=1)

    def test_trainer_with_multiple_metrics(self):
        """Test trainer with multiple evaluation metrics."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model)

        metrics = [
            lambda x, y: np.mean(np.abs(x.numpy() - y.numpy())),
            lambda x, y: np.mean(np.square(x.numpy() - y.numpy())),
        ]

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, eval_metric=metrics, epochs=1)

    def test_trainer_early_stopping(self):
        """Test early stopping functionality."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model)

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
        trainer = Trainer(model)

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, max_grad_norm=1.0, epochs=1)

    def test_trainer_custom_loss(self):
        """Test trainer with custom loss function."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model)

        custom_loss = tf.keras.losses.MeanAbsoluteError()

        trainer.train(train_loader=self.train_loader, valid_loader=self.valid_loader, loss_fn=custom_loss, epochs=1)

    def test_trainer_2gpu(self):
        strategy = tf.distribute.MirroredStrategy()
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model, strategy=strategy)
        trainer.train(self.train_loader, self.valid_loader, **self.fit_config)

    def test_trainer_kwargs(self):
        """Test that custom kwargs are set as attributes."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = Trainer(model, custom_param="test_value", another_param=42)

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

        trainer = KerasTrainer(
            model,
        )
        trainer.train(
            train_dataset=(x_train, y_train),
            valid_dataset=(x_valid, y_valid),
            optimizer=tf.keras.optimizers.Adam(0.003),
            **self.fit_config
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
        trainer = KerasTrainer(
            model,
        )
        trainer.train(train_loader, valid_loader, optimizer=tf.keras.optimizers.Adam(0.003), **self.fit_config)
        trainer.save_model("./weights")

    def test_trainer_fit_alias(self):
        """Test that fit() is an alias for train()."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        history = trainer.fit(train_dataset=(x_train, y_train), epochs=1, batch_size=1)
        self.assertIsNotNone(history)

    def test_trainer_with_string_optimizer(self):
        """Test training with optimizer specified as string."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        trainer.train(train_dataset=(x_train, y_train), optimizer="adam", epochs=1, batch_size=1)

    def test_trainer_with_dict_optimizer(self):
        """Test training with optimizer specified as dict."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

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
        trainer = KerasTrainer(model)

        trainer.train(train_dataset=(x_train, y_train), loss_fn="mae", epochs=1, batch_size=1)

    def test_trainer_with_metrics(self):
        """Test training with metrics."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        trainer.train(train_dataset=(x_train, y_train), metrics=["mae", "mse"], epochs=1, batch_size=1)

    def test_trainer_with_callbacks(self):
        """Test training with custom callbacks."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)

        trainer.train(train_dataset=(x_train, y_train), callbacks=[early_stopping], epochs=5, batch_size=1)

    def test_trainer_with_steps_per_epoch(self):
        """Test training with custom steps_per_epoch."""
        x_train = np.random.random((10, 10, 1))
        y_train = np.random.randint(0, 2, (10, 2, 1))
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(2)

        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        trainer.train(train_dataset=train_loader, steps_per_epoch=2, epochs=1)

    def test_trainer_run_eagerly(self):
        """Test training with run_eagerly=True."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        trainer.train(train_dataset=(x_train, y_train), run_eagerly=True, epochs=1, batch_size=1)

    def test_trainer_verbose_levels(self):
        """Test training with different verbose levels."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        for verbose in [0, 1, 2]:
            trainer = KerasTrainer(model)
            trainer.train(train_dataset=(x_train, y_train), verbose=verbose, epochs=1, batch_size=1)

    def test_get_model(self):
        """Test get_model() returns the correct model."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)

        retrieved_model = trainer.get_model()
        self.assertIsInstance(retrieved_model, tf.keras.Model)

    def test_plot(self):
        """Test plot functionality."""
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)
        trainer = KerasTrainer(model)

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

        trainer = KerasTrainer(keras_model)
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
        trainer = Seq2seqKerasTrainer(model)

        self.assertIsInstance(trainer, KerasTrainer)

    def test_basic_training(self):
        """Test basic training with Seq2seqKerasTrainer."""
        x_train = np.random.random((2, 10, 1))
        y_train = np.random.randint(0, 2, (2, 2, 1))
        config = AutoConfig.for_model("rnn")
        model = AutoModel.from_config(config, predict_sequence_length=2)

        trainer = Seq2seqKerasTrainer(model)
        trainer.train(train_dataset=(x_train, y_train), epochs=1, batch_size=1)


if __name__ == "__main__":
    unittest.main()
