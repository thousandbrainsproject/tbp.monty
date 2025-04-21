# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import dataclasses
import importlib
from inspect import Parameter, signature
from typing import Callable, Optional, Type

__all__ = [
    "as_dataclass_dict",
    "create_dataclass_args",
    "from_dataclass_dict",
]

# Keeps track of the dataclass type in a serializable dataclass dict
_DATACLASS_TYPE = "__dataclass_type__"


def as_dataclass_dict(obj):
    """Convert a dataclass instance to a serializable dataclass dict.

    Args:
        obj: The dataclass instance to convert

    Returns:
        A dictionary with the dataclass fields and values

    Raises:
        TypeError: If the object is not a dataclass instance
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"Expecting dataclass instance but got {type(obj)}")

    result = {_DATACLASS_TYPE: f"{obj.__module__}.{obj.__class__.__name__}"}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        # Handle nested dataclassess
        if dataclasses.is_dataclass(value):
            value = as_dataclass_dict(value)
        result[f.name] = value
    return result


def from_dataclass_dict(datadict):
    """Convert a serializable dataclass dict back into the original dataclass.

    Expecting that the serializable dataclass dict was created via :func:`.asdict`.

    Args:
        datadict: The serializable dataclass dict to convert

    Returns:
        The original dataclass instance

    Raises:
        TypeError: If the object is not a dict instance
    """
    if not isinstance(datadict, dict):
        raise TypeError(f"Expecting dict instance but got {type(datadict)}")

    # Check for nested dataclass
    kwargs = {}
    for k, v in datadict.items():
        kwargs[k] = from_dataclass_dict(v) if isinstance(v, dict) else v

    if _DATACLASS_TYPE not in kwargs:
        # Not a dataclass dict
        return kwargs

    # Get dataclass module and type
    module_name, class_name = kwargs.pop(_DATACLASS_TYPE).rsplit(".", 1)

    # Load dataclass module and recreate the instance
    module = importlib.import_module(module_name)
    dataclass_type = getattr(module, class_name)

    return dataclass_type(**kwargs)


def extract_fields(function):
    # Extract function signature
    sig = signature(function)

    _fields = []
    # Convert function parameters to dataclass fields
    for p in sig.parameters.values():
        # Ignore "self" in case of class methods
        if p.name == "self":
            continue
        f = [p.name]
        if p.default != Parameter.empty:
            # Infer data type from default value
            if p.annotation != Parameter.empty:
                t = type(p.default)
            else:
                t = p.annotation
            f.extend([t, dataclasses.field(default=p.default)])
        elif p.annotation != Parameter.empty:
            f.append(p.annotation)
        _fields.append(f)

    return _fields


def create_dataclass_args(
    dataclass_name: str,
    function: Callable,
    base: Optional[Type] = None,
):
    """Creates configuration dataclass args from a given function arguments.

    When the function arguments have type annotations these annotations will be
    passed to the dataclass fields, otherwise the type will be inferred from the
    argument default value, if any.

    For example::

        SingleSensorAgentArgs = create_dataclass_args(
                "SingleSensorAgentArgs", SingleSensorAgent.__init__)

        # Is equivalent to
        @dataclass(frozen=True)
        class SingleSensorAgentArgs:
            agent_id: str
            sensor_id: str
            position: Tuple[float. float, float] = (0.0, 1.5, 0.0)
            rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
            height: float = 0.0
            :

    Args:
        dataclass_name: The name of the new dataclass
        function: The function used to extract the parameters for the dataclass
        base: Optional base class for newly created dataclass

    Returns:
        New dataclass with fields defined by the function arguments.
    """
    _fields = extract_fields(function)

    # Add base class to new dataclass if given. Limited to a single base class
    bases = (base,) if base is not None else ()
    return dataclasses.make_dataclass(dataclass_name, _fields, bases=bases, frozen=True)


def config_to_dict(config):
    """Convert config composed of mixed dataclass and dict elements to pure dict.

    We want to convert configs composed of mixed dataclass and dict elements to
    pure dicts without dataclasses for backward compatibility.

    TODO: Remove once all other configs are converted to dict only

    Returns:
        Pure dict version of config.
    """
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return config


def get_subset_of_args(arguments, function):
    dict_args = config_to_dict(arguments)
    _fields = extract_fields(function)
    common_fields = {}
    for field in _fields:
        field_name = field[0]
        if field_name in dict_args:
            common_fields[field_name] = dict_args[field_name]

    return common_fields
