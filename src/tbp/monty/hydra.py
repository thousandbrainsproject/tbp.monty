# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tbp.monty.frameworks.agents import AgentID


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


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("monty.agent_id", agent_id_resolver)
    OmegaConf.register_new_resolver("monty.class", monty_class_resolver)
    OmegaConf.register_new_resolver("np.array", ndarray_resolver)
    OmegaConf.register_new_resolver("np.ones", ones_resolver)
    OmegaConf.register_new_resolver("np.list_eval", numpy_list_eval_resolver)
    OmegaConf.register_new_resolver("path.expanduser", path_expanduser_resolver)
