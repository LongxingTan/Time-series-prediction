import logging
from typing import Any, Callable, Dict, List, Optional

import tensorflow as tf

from ..models import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class Pipeline(object):
    _load_processor = False

    def __init__(self, cfg, processor: Optional[Callable] = None):
        self.cfg = cfg
        self.backbone = None
        self.model = None
        self.label_scaler = None
        self.processor = processor

    def build_model(self, n_features, n_outputs):
        # Update model config with actual data dimensions
        self.cfg.model.n_features = n_features
        self.cfg.model.n_outputs = n_outputs

        config = AutoConfig()(self.cfg.model_name)
        config.output_size = n_outputs

        self._model = AutoModel.from_config(
            config=config, predict_sequence_length=self.cfg.model.predict_sequence_length
        )

        inputs = tf.keras.Input(shape=(self.cfg.model.train_sequence_length, self.cfg.model.n_features))
        outputs = self._model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # loss_fn = getattr(tf.keras.losses, self.cfg.training.loss)(reduction=tf.keras.losses.Reduction.SUM)
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        optimizer = getattr(tf.keras.optimizers, self.cfg.training.optimizer)(
            learning_rate=self.cfg.training.learning_rate
        )
        # metrics = [getattr(tf.keras.metrics, m)() for m in self.cfg.training.metrics]
        model.compile(loss=loss_fn, optimizer=optimizer)  # metrics=metrics
        return model

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        sample_x, sample_y = train_dataset[0]
        n_features = sample_x.shape[-1]
        n_outputs = sample_y.shape[-1]
        self.model = self.build_model(n_features, n_outputs)

        print(f"Training with {n_features} features and {n_outputs} outputs.")
        print(self.model.summary())

        history = self.model.fit(
            train_dataset, validation_data=eval_dataset, epochs=self.cfg.training.epochs, callbacks=callbacks, verbose=1
        )
        return history

    def predict(self, test_dataset, save_path):
        # Ensure model is built and loaded if not already trained in this session
        if self.model is None:
            # Need to get n_features and n_outputs from the test data itself
            sample_x, id = test_dataset[0]
            n_features = sample_x.shape[-1]
            n_outputs = self.cfg.model.n_outputs  # Use configured output if model wasn't trained
            self.model = self.build_model(n_features, n_outputs)
            self.model.load_weights("/weights")

    def get_iterator(self):
        pass

    def postprocess(self):
        pass
