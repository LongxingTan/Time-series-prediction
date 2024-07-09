from abc import ABC, abstractmethod
import collections
import json
import os
from typing import Any, Dict, Union


class BaseModel(ABC):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, name: str):
        return

    @classmethod
    def from_pretrained(cls, name: str):
        return


class BaseConfig(ABC):
    def __init__(self, **kwargs):
        self.update(kwargs)

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def to_dict(self):
        # output_dict = {}
        # for key, value in self.__dict__.items():
        #     output_dict[key] = value
        # return flatten_dict(output_dict)
        return {key: getattr(self, key) for key in self.__dict__}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        with open(pretrained_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_pretrained(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat
