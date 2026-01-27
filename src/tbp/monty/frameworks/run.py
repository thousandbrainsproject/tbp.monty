# Copyright 2025-2026 Thousand Brains Project
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


def output_dir_from_run_name(config: DictConfig) -> Path:
    """Configure the output directory unique to the run name.

    The output directory is created if it does not exist.

    Args:
        config: Hydra config.

    Returns:
        output_dir: Path to run name-specific output directory.
    """
    output_dir = (
        Path(config.experiment.config.logging.output_dir)
        / config.experiment.config.logging.run_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


@hydra.main(config_path="../../../conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    print_config(cfg)
    register_resolvers()

    cfg.experiment.config.logging.output_dir = str(output_dir_from_run_name(cfg))

    experiment = hydra.utils.instantiate(cfg.experiment)
    start_time = time.time()
    with experiment:
        experiment.run()

    logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")
