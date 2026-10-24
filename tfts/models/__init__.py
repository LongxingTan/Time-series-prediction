"""Public exports, loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "AutoConfig": ("tfts.models.auto_config", "AutoConfig"),
    "BaseConfig": ("tfts.config", "BaseConfig"),
    "BaseModel": ("tfts.models.base", "BaseModel"),
    "CommonConfig": ("tfts.config", "CommonConfig"),
    "get_model_capabilities": ("tfts.models.catalog", "get_model_capabilities"),
    "get_model_info": ("tfts.models.catalog", "get_model_info"),
    "list_models": ("tfts.models.catalog", "list_models"),
    "list_supported_tasks": ("tfts.models.catalog", "list_supported_tasks"),
    "register_model": ("tfts.models.catalog", "register_model"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
