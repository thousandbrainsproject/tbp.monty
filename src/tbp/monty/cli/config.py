# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from tbp.monty.conf import Config
from tbp.monty.core.experiment import ExperimentConfig
from tbp.monty.platform import PlatformConfig

__all__ = ["config"]


@dataclass(frozen=True)
class Environment:
    """Add your environment variables (e.g. for config file interpolation) here."""

    MONTY_LOGS: str = "~/tbp/results/monty"
    MONTY_MODELS: str = "~/tbp/results/monty/pretrained_models"
    MONTY_DATA: str = "~/tbp/data"
    WANDB_DIR: str = "~/tbp/results/monty"

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not value:
                value = str(field.default)
                print(f"{field.name} not set. Using default: {value}")
            if value.startswith("~/"):
                value = str(Path(value).expanduser().resolve())

            os.environ[field.name] = value


def config(args: dict[str, Any]) -> tuple[PlatformConfig, ExperimentConfig | None]:
    Environment(
        MONTY_LOGS=args["monty.logs_dir"],
        MONTY_MODELS=args["monty.models_dir"],
        MONTY_DATA=args["monty.data_dir"],
        WANDB_DIR=args["wandb.dir"],
    )

    yaml_configuration = Config.get(args)

    return PlatformConfig(yaml_configuration), yaml_configuration
