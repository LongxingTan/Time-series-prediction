"""Utilities for loading Keras models that contain TFTS objects."""

import importlib
import inspect
import json
import os
from typing import Any, Dict, Iterator, Mapping, Optional, Union
import zipfile

import tensorflow as tf

PathLike = Union[str, os.PathLike]


def _walk_config(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_config(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_config(child)


def get_custom_objects(filepath: PathLike) -> Dict[str, Any]:
    """Discover TFTS objects referenced by a Keras v3 ``.keras`` archive.

    Only modules inside the installed :mod:`tfts` package are imported. This
    keeps model loading convenient without importing arbitrary modules named in
    an untrusted model archive.
    """
    filepath = os.fspath(filepath)
    if not zipfile.is_zipfile(filepath):
        return {}

    with zipfile.ZipFile(filepath) as archive:
        try:
            config = json.loads(archive.read("config.json"))
        except KeyError as exc:
            raise ValueError(f"{filepath!r} is missing Keras config.json") from exc

    objects: Dict[str, Any] = {}
    for item in _walk_config(config):
        module_name = item.get("module")
        class_name = item.get("class_name")
        if not (
            isinstance(module_name, str)
            and (module_name == "tfts" or module_name.startswith("tfts."))
            and isinstance(class_name, str)
        ):
            continue

        module = importlib.import_module(module_name)
        try:
            obj = getattr(module, class_name)
        except AttributeError as exc:
            raise TypeError(f"Could not find {class_name!r} in {module_name!r}") from exc

        names = {class_name}
        registered_name = item.get("registered_name")
        if isinstance(registered_name, str):
            names.add(registered_name)

        for name in names:
            previous = objects.get(name)
            if previous is not None and previous is not obj:
                raise ValueError(
                    f"The model contains multiple TFTS objects named {name!r}. "
                    "Pass the intended class explicitly through custom_objects."
                )
            objects[name] = obj

    return objects


def load_model(
    filepath: PathLike,
    custom_objects: Optional[Mapping[str, Any]] = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: Any,
) -> tf.keras.Model:
    """Load a Keras model and automatically resolve its TFTS custom objects.

    This is a drop-in convenience wrapper around
    :func:`tf.keras.models.load_model`. Explicit ``custom_objects`` take
    precedence over objects discovered from the archive.
    """
    discovered_objects = get_custom_objects(filepath)
    if custom_objects:
        discovered_objects.update(custom_objects)

    load_kwargs = {
        "custom_objects": discovered_objects,
        "compile": compile,
        **kwargs,
    }
    if "safe_mode" in inspect.signature(tf.keras.models.load_model).parameters:
        load_kwargs["safe_mode"] = safe_mode

    return tf.keras.models.load_model(filepath, **load_kwargs)
