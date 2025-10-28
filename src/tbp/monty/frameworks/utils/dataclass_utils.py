# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import dataclasses
import importlib
from inspect import Parameter, signature
from typing import Any, Callable, ClassVar, Dict, Protocol, Type

from typing_extensions import TypeIs


class Dataclass(Protocol):
    """A protocol for dataclasses to be used in type hints.

    The reason this exists is because dataclass.dataclass is not a valid type.
    """

    __dataclass_fields__: ClassVar[Dict[str, Any]]
    """Checking for presence of __dataclass_fields__ is a hack to check if a class is a
    dataclass."""


__all__ = [
    "as_dataclass_dict",
    "create_dataclass_args",
    "from_dataclass_dict",
    "Dataclass",
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
    base: Type | None = None,
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
            agent_id: AgentID
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


def config_to_dict(config: Dataclass | Dict[str, Any]) -> Dict[str, Any]:
    """Convert config composed of mixed dataclass and dict elements to pure dict.

    We want to convert configs composed of mixed dataclass and dict elements to
    pure dicts without dataclasses for backward compatibility.

    Like `dataclasses.asdict` (and `dataclasses._asdict_inner`), objects that
    are not or do not contain dataclass instances are deep-copied and returned.

    Args:
        config: dict or dataclass instance to convert to dict.

    Returns:
        Pure dict version of config.

    Raises:
        TypeError: If the object is not a dict or dataclass instance
    """
    if is_config_like(config):
        return _config_to_dict_inner(config)
    else:
        msg = f"Expecting dict or dataclass instance but got {type(config)}"
        raise TypeError(msg)


def _config_to_dict_inner(obj: Any) -> Any:
    """Recursively convert any dataclass instances to dictionaries.

    This function is used to convert dataclass instances to dictionaries, including
    any dataclass instances nested within dictionaries, lists, tuples
    (including namedtuples), and dataclass fields. It replicates
    `dataclasses._asdict_inner`. It is reimplemented here
    since `dataclasses._asdict_inner` is not public.

    Args:
        obj: Any object that may be a dataclass instance or contain one.

    Returns:
        Like `obj` but with any dataclass instances converted to dictionaries.
    """
    if is_dataclass_instance(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = _config_to_dict_inner(getattr(obj, f.name))
            result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.
        return type(obj)(*[_config_to_dict_inner(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_config_to_dict_inner(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_config_to_dict_inner(k), _config_to_dict_inner(v)) for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


def is_config_like(obj: Any) -> TypeIs[Dataclass | Dict[str, Any]]:
    """Returns True if obj is a dataclass or dict, False otherwise.

    Args:
        obj: Object to check.

    Returns:
        True if config is a dataclass or dict, False otherwise.
    """
    return isinstance(obj, dict) or is_dataclass_instance(obj)


def is_dataclass_instance(obj: Any) -> bool:
    """Returns True if obj is an instance of a dataclass.

    This function replicates `dataclasses._is_dataclass_instance`.  It is
    reimplemented here since `dataclasses._is_dataclass_instance` is not public.

    Args:
        obj: The object to check.

    Returns:
        True if obj is an instance of a dataclass, False otherwise.
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def get_subset_of_args(arguments, function):
    dict_args = config_to_dict(arguments)
    _fields = extract_fields(function)
    common_fields = {}
    for field in _fields:
        field_name = field[0]
        if field_name in dict_args:
            common_fields[field_name] = dict_args[field_name]

    return common_fields
