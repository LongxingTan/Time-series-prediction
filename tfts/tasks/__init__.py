"""Public exports, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "AbsoluteErrorScorer": ("tfts.tasks.anomaly", "AbsoluteErrorScorer"),
    "AnomalyScorer": ("tfts.tasks.anomaly", "AnomalyScorer"),
    "QuantileCalibrator": ("tfts.tasks.anomaly", "QuantileCalibrator"),
    "SquaredErrorScorer": ("tfts.tasks.anomaly", "SquaredErrorScorer"),
    "ClassificationHead": ("tfts.tasks.auto_task", "ClassificationHead"),
    "DistributionForecastHead": ("tfts.tasks.auto_task", "DistributionForecastHead"),
    "PointForecastHead": ("tfts.tasks.auto_task", "PointForecastHead"),
    "QuantileForecastHead": ("tfts.tasks.auto_task", "QuantileForecastHead"),
    "ReconstructionHead": ("tfts.tasks.auto_task", "ReconstructionHead"),
    "BaseHead": ("tfts.tasks.base", "BaseHead"),
    "BaseTask": ("tfts.tasks.base", "BaseTask"),
    "ModelOutput": ("tfts.tasks.base", "ModelOutput"),
    "TimeSeriesTaskModel": ("tfts.tasks.base", "TimeSeriesTaskModel"),
    "Objective": ("tfts.tasks.objectives", "Objective"),
    "AnomalyDetectionModel": ("tfts.tasks.anomaly", "AnomalyDetectionModel"),
    "ClassificationModel": ("tfts.tasks.classification", "ClassificationModel"),
    "ForecastingModel": ("tfts.tasks.forecasting", "ForecastingModel"),
    "ImputationModel": ("tfts.tasks.imputation", "ImputationModel"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
