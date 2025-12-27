from abc import ABC, abstractmethod
from collections import OrderedDict, UserDict
from dataclasses import dataclass, fields
from typing import Any


class BaseTask(ABC):
    """Base task for tfts task."""


class ModelOutput(OrderedDict):
    def __post_init__(self):
        # Automatically populate the OrderedDict with dataclass fields
        class_fields = fields(self)
        for field in class_fields:
            value = getattr(self, field.name)
            if value is not None:
                self[field.name] = value

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.to_tuple()[k]
        return super().__getitem__(k)

    def to_tuple(self) -> tuple[Any]:
        return tuple(getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None)
