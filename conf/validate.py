# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import (
    agent_id_resolver,
    monty_class_resolver,
    ndarray_resolver,
    numpy_list_eval_resolver,
    ones_resolver,
)


@hydra.main(config_path=".", config_name="experiment", version_base=None)
def validate(cfg: DictConfig):
    # Force interpolation of the config to get errors
    OmegaConf.to_object(cfg)
    print(OmegaConf.to_yaml(cfg))

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    app = hydra.utils.instantiate(cfg.experiment)
    with app:
        # exercise .setup_experiment method
        pass

    print("done")


if __name__ == "__main__":
    setup_env()
    OmegaConf.register_new_resolver("monty.agent_id", agent_id_resolver)
    OmegaConf.register_new_resolver("monty.class", monty_class_resolver)
    OmegaConf.register_new_resolver("np.array", ndarray_resolver)
    OmegaConf.register_new_resolver("np.ones", ones_resolver)
    OmegaConf.register_new_resolver("np.list_eval", numpy_list_eval_resolver)
    validate()
