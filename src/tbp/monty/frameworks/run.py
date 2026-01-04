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
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.hydra import register_resolvers

logger = logging.getLogger(__name__)


def print_config(config: DictConfig) -> None:
    """Print config with nice formatting."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(OmegaConf.to_yaml(config))
    print("-" * 100)


@hydra.main(config_path="../../../conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    print_config(cfg)
    register_resolvers()

    output_dir = (
        Path(cfg.experiment.config.logging.output_dir)
        / cfg.experiment.config.logging.run_name
    )
    cfg.experiment.config.logging.output_dir = str(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    experiment = hydra.utils.instantiate(cfg.experiment)
    start_time = time.time()
    with experiment:
        experiment.run()

    logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")
