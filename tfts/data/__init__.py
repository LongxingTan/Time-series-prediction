"""Public exports, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "get_air_passengers": ("tfts.datasets.prepared", "get_air_passengers"),
    "get_data": ("tfts.datasets.prepared", "get_data"),
    "get_sine": ("tfts.datasets.prepared", "get_sine"),
    "SequenceMaterializer": ("tfts.data.tf_dataset", "SequenceMaterializer"),
    "TabularBatch": ("tfts.data.tf_dataset", "TabularBatch"),
    "TabularMaterializer": ("tfts.data.tf_dataset", "TabularMaterializer"),
    "generate_sequence_windows": ("tfts.data.windows", "generate_sequence_windows"),
    "pad_sequence": ("tfts.data.padding", "pad_sequence"),
    "pad_sequences": ("tfts.data.padding", "pad_sequences"),
    "sequence_mask": ("tfts.data.padding", "sequence_mask"),
    "final_windows": ("tfts.data.windows", "final_windows"),
    "sampled_windows": ("tfts.data.windows", "sampled_windows"),
    "Window": ("tfts.data.windows", "Window"),
    "WindowIndex": ("tfts.data.windows", "WindowIndex"),
    "WindowIndexer": ("tfts.data.windows", "WindowIndexer"),
    "WindowSpec": ("tfts.data.windows", "WindowSpec"),
    "Schema": ("tfts.data.schema", "Schema"),
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
