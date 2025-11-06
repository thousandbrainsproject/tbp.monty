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
import logging
import os
import pprint
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.agents import AgentID

logger = logging.getLogger(__name__)


def print_config(config):
    """Print config with nice formatting if config_args.print_config is True."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(pprint.pformat(config))
    print("-" * 100)


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
    return [eval(item) for item in expr_list]  # noqa: S307


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("monty.agent_id", agent_id_resolver)
    OmegaConf.register_new_resolver("monty.class", monty_class_resolver)
    OmegaConf.register_new_resolver("np.array", ndarray_resolver)
    OmegaConf.register_new_resolver("np.ones", ones_resolver)
    OmegaConf.register_new_resolver("np.list_eval", numpy_list_eval_resolver)


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    print_config(cfg)
    register_resolvers()

    os.makedirs(cfg.experiment.config.logging.output_dir, exist_ok=True)
    experiment = hydra.utils.instantiate(cfg.experiment)
    start_time = time.time()
    with experiment as exp:
        # TODO: Later will want to evaluate every x episodes or epochs
        # this could probably be solved with just setting the logging freqency
        # Since each trainng loop already does everything that eval does.
        if exp.do_train:
            print("---------training---------")
            exp.train()

        if exp.do_eval:
            print("---------evaluating---------")
            exp.evaluate()
    logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")
