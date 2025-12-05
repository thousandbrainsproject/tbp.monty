# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import os
import pprint
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.hydra import register_resolvers

logger = logging.getLogger(__name__)


def print_config(config):
    """Print config with nice formatting if config_args.print_config is True."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(pprint.pformat(config))
    print("-" * 100)


@hydra.main(config_path="../../../conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    if cfg.config.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    register_resolvers()
    print(OmegaConf.to_yaml(cfg))

    output_dir = (
        Path(cfg.config.logging.output_dir)
        / cfg.config.logging.run_name
    )
    cfg.config.logging.output_dir = str(output_dir)

    os.makedirs(cfg.config.logging.output_dir, exist_ok=True)
    experiment = hydra.utils.instantiate(cfg)
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
