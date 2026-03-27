# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID


def agent_id_resolver(agent_id: str) -> AgentID:
    """Returns an AgentID new type from a string."""
    return AgentID(agent_id)


def monty_class_resolver(class_name: str) -> type:
    """Returns a class object by fully qualified path.

    TODO: This is an interim solution to retrieve my_class in
      the my_class(**my_args) pattern.
    """
    parts = class_name.split(".")
    module = ".".join(parts[:-1])
    klass = parts[-1]
    module_obj = importlib.import_module(module)
    return getattr(module_obj, klass)


def ndarray_resolver(list_or_tuple: list | tuple) -> np.ndarray:
    """Returns a numpy array from a list or tuple."""
    return np.array(list_or_tuple)


def ones_resolver(n: int) -> np.ndarray:
    """Returns a numpy array of ones."""
    return np.ones(n)


def numpy_list_eval_resolver(expr_list: list) -> list[float]:
    # call str() on each item so we can use number literals
    return [eval(str(item)) for item in expr_list]  # noqa: S307


def path_expanduser_resolver(path: str) -> str:
    """Returns a path with ~ expanded to the user's home directory."""
    return str(Path(path).expanduser())


def sensor_id_resolver(sensor_id: str) -> SensorID:
    """Returns a SensorID new type from a string."""
    return SensorID(sensor_id)


def tests_dir_resolver(path: str) -> str:
    return str(Path(__file__).parents[3] / "tests" / Path(path))


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers for Monty configs.

    Skips resolvers that are already registered rather than raising
    a ValueError, since multiple entry points (e.g. tests/__init__.py
    and update_snapshots.py) may call this function in the same process.
    """
    resolvers: dict[str, Callable[..., Any]] = {
        "monty.agent_id": agent_id_resolver,
        "monty.class": monty_class_resolver,
        "monty.sensor_id": sensor_id_resolver,
        "np.array": ndarray_resolver,
        "np.ones": ones_resolver,
        "np.list_eval": numpy_list_eval_resolver,
        "path.expanduser": path_expanduser_resolver,
        "path.tests": tests_dir_resolver,
    }
    for name, resolver in resolvers.items():
        with contextlib.suppress(ValueError):
            OmegaConf.register_new_resolver(name, resolver)
