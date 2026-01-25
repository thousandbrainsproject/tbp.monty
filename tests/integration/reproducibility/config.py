# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path


def hydra_config(
    test_name: str,
    output_dir: Path,
    fixed_actions_path: Path | None = None,
    model_name_or_path: Path | None = None,
) -> DictConfig:
    """Helper for composing a Hydra configuration.

    Args:
        test_name: The name of the test to run. Must be a present in the
            conf/experiment/test directory.
        output_dir: The directory to store the output.
        fixed_actions_path: The path to the fixed actions file.
        model_name_or_path: The path to the model to load.

    Returns:
        The composed Hydra configuration.
    """
    overrides = [
        f"experiment=test/{test_name}",
        "num_parallel=1",
        f"++experiment.config.logging.output_dir={output_dir}",
    ]
    if fixed_actions_path:
        overrides.append(
            "+experiment.config.monty_config.motor_system_config"
            f".motor_system_args.policy_args.file_name={fixed_actions_path}",
        )
    if model_name_or_path:
        overrides.append(
            f"experiment.config.model_name_or_path={model_name_or_path}",
        )

    return hydra.compose(config_name="experiment", overrides=overrides)
