import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from ..models import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class Pipeline(object):
    _load_processor = False

    def __init__(self, cfg, processor: Optional[Callable] = None, strategy: Optional[tf.distribute.Strategy] = None):
        self.cfg = cfg
        self.backbone = None
        self.model = None
        self.label_scaler = None
        self.processor = processor
        self.strategy = strategy or self._setup_strategy()

    def _setup_strategy(self):
        """Detects GPUs and returns the appropriate distribution strategy."""
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            logger.info("Using OneDeviceStrategy (1 GPU)")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            logger.info("Using default strategy (CPU)")
            return tf.distribute.get_strategy()

    def build_model(self, n_features, n_outputs):
        # Update model config with actual data dimensions
        with self.strategy.scope():
            self.cfg.model.n_features = n_features
            self.cfg.model.n_outputs = n_outputs

            config = AutoConfig()(self.cfg.model.name)
            config.output_size = n_outputs

            self._model = AutoModel.from_config(
                config=config, predict_sequence_length=self.cfg.model.predict_sequence_length
            )

            inputs = tf.keras.Input(shape=(self.cfg.model.train_sequence_length, self.cfg.model.n_features))
            outputs = self._model(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            loss_fn = getattr(tf.keras.losses, self.cfg.training.loss)()

            optimizer = getattr(tf.keras.optimizers, self.cfg.training.optimizer)(
                learning_rate=self.cfg.training.learning_rate
            )
            # metrics = [getattr(tf.keras.metrics, metric)() for metric in self.cfg.training.metrics]
            model.compile(loss=loss_fn, optimizer=optimizer)  # metrics=metrics
        return model

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        try:
            sample_x, sample_y = next(iter(train_dataset))
        except (TypeError, StopIteration, AttributeError):
            sample_x, sample_y = train_dataset[0]

        n_features = sample_x.shape[-1]
        n_outputs = sample_y.shape[-1]
        self.model = self.build_model(n_features, n_outputs)

        print(f"Training with {n_features} features and {n_outputs} outputs.")
        if self.strategy.num_replicas_in_sync > 1:
            print(f"Distributing training across {self.strategy.num_replicas_in_sync} replicas.")
        print(self.model.summary())

        history = self.model.fit(
            train_dataset, validation_data=eval_dataset, epochs=self.cfg.training.epochs, callbacks=callbacks, verbose=1
        )
        return history

    def predict(self, test_dataset, weights_path=None):
        # Ensure model is built and loaded if not already trained in this session
        if self.model is None:
            # Need to get n_features and n_outputs from the test data itself
            sample_x, id = test_dataset[0]
            n_features = sample_x.shape[-1]
            n_outputs = self.cfg.model.n_outputs  # Use configured output if model wasn't trained
            self.model = self.build_model(n_features, n_outputs)
            if weights_path and os.path.exists(weights_path):
                self.model.load_weights(weights_path)

        return self.model.predict(test_dataset)

    def generate(
        self, initial_sequence: Union[np.ndarray, tf.Tensor], horizon: int, extra_features_fn: Optional[Callable] = None
    ):
        """
        Auto-regressive generation for multi-step time series.
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before generation.")

        current_window = tf.convert_to_tensor(initial_sequence, dtype=tf.float32)
        batch_size = tf.shape(current_window)[0]
        train_len = self.cfg.model.train_sequence_length
        output_len = self.cfg.model.predict_sequence_length
        n_features = tf.shape(current_window)[-1]

        all_predictions = []
        steps_needed = int(np.ceil(horizon / output_len))

        print(f"Generating {horizon} steps in {steps_needed} blocks...")

        for step in range(steps_needed):
            model_input = current_window[:, -train_len:, :]

            # 2. Predict next chunk (Batch, Output_Len)
            preds = self.model(model_input, training=False)

            if len(preds.shape) == 2:
                preds = tf.expand_dims(preds, axis=-1)

            # 3. Prepare features for the predicted chunk (Recursive Step)
            # If your model uses (Value, Mask), predicted values get Mask=0
            if n_features > 1:
                # Create a zero-mask for the new predictions
                # Adjust this logic if you have other features like 'Day of Year'
                padding_features = tf.zeros((batch_size, output_len, n_features - 1))
                new_chunk = tf.concat([preds, padding_features], axis=-1)
            else:
                new_chunk = preds

            # 4. Concatenate back to context
            current_window = tf.concat([current_window, new_chunk], axis=1)
            all_predictions.append(preds)

        # 5. Post-process the full sequence, Concatenate chunks and trim to exact horizon
        full_forecast = tf.concat(all_predictions, axis=1)
        return full_forecast[:, :horizon, :]

    def postprocess(self):
        pass
