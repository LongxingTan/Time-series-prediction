"""One registry primitive for every TFTS extension point."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Union


@dataclass(frozen=True)
class ComponentSpec:
    """Serializable reference to a configured registered component."""

    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("component name must be a non-empty string")
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kwargs": dict(self.kwargs)}

    @classmethod
    def from_value(cls, value: Union[str, Mapping[str, Any], "ComponentSpec"]) -> "ComponentSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            return cls(name=value["name"], kwargs=value.get("kwargs", {}))
        raise TypeError("component specification must be a name, mapping, or ComponentSpec")


@dataclass(frozen=True)
class Registration:
    factory: Callable[..., Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, Dict[str, Registration]] = defaultdict(dict)


def register(kind: str, name: str, obj: Optional[Callable[..., Any]] = None, **meta):
    """Register a factory directly or as a decorator."""

    if not isinstance(kind, str) or not kind:
        raise ValueError("registry kind must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("registry name must be a non-empty string")

    def decorator(factory):
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing.factory is not factory:
            raise ValueError(f"{kind} {name!r} is already registered")
        _REGISTRY[kind][name] = Registration(factory=factory, metadata=dict(meta))
        return factory

    return decorator if obj is None else decorator(obj)


def resolve(kind: str, name: str) -> Registration:
    try:
        return _REGISTRY[kind][name]
    except KeyError as error:
        raise KeyError(f"Unknown {kind} {name!r}. Available: {available(kind)}") from error


def available(kind: str) -> List[str]:
    return sorted(_REGISTRY.get(kind, ()))


def build(kind: str, component, **context):
    """Build a named/spec component, or return an already-live object."""

    if isinstance(component, (str, ComponentSpec, Mapping)):
        spec = ComponentSpec.from_value(component)
        registration = resolve(kind, spec.name)
        kwargs = {**spec.kwargs, **context}
        return registration.factory(**kwargs)
    return component


register_model = partial(register, "model")
register_decoder = partial(register, "decoder")
register_processor = partial(register, "processor")
register_objective = partial(register, "objective")
register_task = partial(register, "task")
register_transform = partial(register, "transform")


__all__ = [
    "ComponentSpec",
    "Registration",
    "available",
    "build",
    "register",
    "register_decoder",
    "register_model",
    "register_objective",
    "register_processor",
    "register_task",
    "register_transform",
    "resolve",
]
