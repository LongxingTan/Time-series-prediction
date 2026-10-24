"""TFTS public API. Implementations are loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    "AnomalyDetectionTaskConfig": ("tfts.tasks.anomaly", "AnomalyDetectionTaskConfig"),
    "ClassificationTaskConfig": ("tfts.tasks.classification", "ClassificationTaskConfig"),
    "ForecastTaskConfig": ("tfts.tasks.forecasting", "ForecastTaskConfig"),
    "GraphStructure": ("tfts.graph.structure", "GraphStructure"),
    "SpatialArrangement": ("tfts.graph.structure", "SpatialArrangement"),
    "SpatialStructure": ("tfts.graph.structure", "SpatialStructure"),
    "TopologyInput": ("tfts.graph.structure", "TopologyInput"),
    "ImputationTaskConfig": ("tfts.tasks.imputation", "ImputationTaskConfig"),
    "SequenceRegressionTaskConfig": ("tfts.tasks.sequence_regression", "SequenceRegressionTaskConfig"),
    "TaskType": ("tfts.tasks.base", "TaskType"),
    "TimeSeriesBatch": ("tfts.batch", "TimeSeriesBatch"),
    "SequenceMaterializer": ("tfts.data", "SequenceMaterializer"),
    "TabularMaterializer": ("tfts.data", "TabularMaterializer"),
    "WindowIndexer": ("tfts.data", "WindowIndexer"),
    "WindowSpec": ("tfts.data", "WindowSpec"),
    "generate_sequence_windows": ("tfts.data", "generate_sequence_windows"),
    "get_data": ("tfts.data", "get_data"),
    "pad_sequence": ("tfts.data", "pad_sequence"),
    "pad_sequences": ("tfts.data", "pad_sequences"),
    "sequence_mask": ("tfts.data", "sequence_mask"),
    "Estimator": ("tfts.estimator", "Estimator"),
    "Forecaster": ("tfts.estimator", "Forecaster"),
    "NotFittedError": ("tfts.estimator", "NotFittedError"),
    "AutoFeatureEngineer": ("tfts.data.features", "AutoFeatureEngineer"),
    "FeaturePipeline": ("tfts.data.features", "FeaturePipeline"),
    "FeatureRole": ("tfts.data.features", "FeatureRole"),
    "FeatureSelection": ("tfts.data.features", "FeatureSelection"),
    "FeatureSpec": ("tfts.data.features", "FeatureSpec"),
    "Schema": ("tfts.data.schema", "Schema"),
    "GenerationConfig": ("tfts.generation", "GenerationConfig"),
    "prepare_generation_batch": ("tfts.generation", "prepare_generation_batch"),
    "from_adjacency": ("tfts.graph", "from_adjacency"),
    "from_correlation": ("tfts.graph", "from_correlation"),
    "from_grid": ("tfts.graph", "from_grid"),
    "from_knn": ("tfts.graph", "from_knn"),
    "from_radius": ("tfts.graph", "from_radius"),
    "evaluate_metrics": ("tfts.metrics", "evaluate"),
    "AutoConfig": ("tfts.models.auto_config", "AutoConfig"),
    "AutoBackbone": ("tfts.tasks.auto", "AutoBackbone"),
    "AutoModel": ("tfts.tasks.auto", "AutoModel"),
    "AutoModelForAnomaly": ("tfts.tasks.auto", "AutoModelForAnomaly"),
    "AutoModelForAnomalyDetection": ("tfts.tasks.auto", "AutoModelForAnomalyDetection"),
    "AutoModelForClassification": ("tfts.tasks.auto", "AutoModelForClassification"),
    "AutoModelForForecasting": ("tfts.tasks.auto", "AutoModelForForecasting"),
    "AutoModelForImputation": ("tfts.tasks.auto", "AutoModelForImputation"),
    "AutoModelForPrediction": ("tfts.tasks.auto", "AutoModelForPrediction"),
    "AutoModelForQuantile": ("tfts.tasks.auto", "AutoModelForQuantile"),
    "AutoModelForTimeSeriesClassification": ("tfts.tasks.auto", "AutoModelForTimeSeriesClassification"),
    "list_models": ("tfts.models.catalog", "list_models"),
    "list_supported_tasks": ("tfts.models.catalog", "list_supported_tasks"),
    "resolve_model_features": ("tfts.estimator.compat", "resolve_model_features"),
    "EagerTrainer": ("tfts.training", "EagerTrainer"),
    "KerasTrainer": ("tfts.training", "KerasTrainer"),
    "Trainer": ("tfts.training", "Trainer"),
    "TrainingArguments": ("tfts.training", "TrainingArguments"),
    "load_model": ("tfts.training", "load_model"),
    "set_seed": ("tfts.training", "set_seed"),
    "OptunaTuner": ("tfts.tuner", "OptunaTuner"),
    "FramePipeline": ("tfts.data.pipeline", "FramePipeline"),
    "TemporalSplit": ("tfts.data.split", "TemporalSplit"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__version__ = "0.0.5"
