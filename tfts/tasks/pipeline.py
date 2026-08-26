"""Task-neutral inference and training pipeline."""

from typing import Any, Callable, Optional

from tfts.contracts import TaskType
from tfts.models.auto_config import AutoConfig
from tfts.models.auto_model import AutoModel
from tfts.trainer import Trainer


class TaskPipeline:
    """Compose preprocessing, a task model, and postprocessing.

    The processor is intentionally a boundary object. It may be a callable or
    expose ``transform``/``inverse_transform``; model and task layers never
    import it.
    """

    def __init__(
        self,
        task,
        model,
        *,
        config=None,
        task_config=None,
        processor: Optional[Callable] = None,
        **task_kwargs,
    ):
        self.task = TaskType.normalize(task)
        self.processor = processor
        if isinstance(model, str):
            config = config or AutoConfig.for_model(model)
            self.model = AutoModel.from_config(config, task=self.task, task_config=task_config, **task_kwargs)
        else:
            if config is not None or task_config is not None or task_kwargs:
                raise ValueError("Config arguments cannot be used with an instantiated model")
            self.model = model
        self.trainer = Trainer(self.model)

    def preprocess(self, inputs):
        if self.processor is None:
            return inputs
        if hasattr(self.processor, "transform"):
            return self.processor.transform(inputs)
        return self.processor(inputs)

    def forward(self, model_inputs, generation_config=None, **kwargs):
        if generation_config is not None:
            if self.task != TaskType.FORECASTING:
                raise ValueError("generation_config is only valid for forecasting")
            return self.model.generate(model_inputs, generation_config, **kwargs)
        if self.task == TaskType.ANOMALY_DETECTION and kwargs.pop("detect", False):
            return self.model.detect(model_inputs)
        return self.model(model_inputs, return_dict=True, training=False)

    def postprocess(self, output):
        if self.processor is None or not hasattr(self.processor, "inverse_transform"):
            return output
        if hasattr(output, "predictions") and output.predictions is not None:
            output.predictions = self.processor.inverse_transform(output.predictions)
            output["predictions"] = output.predictions
        elif hasattr(output, "imputed_values") and output.imputed_values is not None:
            output.imputed_values = self.processor.inverse_transform(output.imputed_values)
            output["imputed_values"] = output.imputed_values
        return output

    def __call__(self, inputs, generation_config=None, **kwargs):
        model_inputs = self.preprocess(inputs)
        output = self.forward(model_inputs, generation_config=generation_config, **kwargs)
        return self.postprocess(output)

    def fit(self, train_dataset, valid_dataset=None, **kwargs):
        return self.trainer.train(train_dataset, valid_dataset=valid_dataset, **kwargs)

    def calibrate(self, inputs):
        if self.task != TaskType.ANOMALY_DETECTION:
            raise ValueError("calibrate() is only valid for anomaly detection")
        return self.model.calibrate(self.preprocess(inputs))


Pipeline = TaskPipeline
