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

import hydra
from omegaconf import DictConfig

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
