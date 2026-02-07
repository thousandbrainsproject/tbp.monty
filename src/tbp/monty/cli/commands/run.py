# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.core.experiment import ExperimentConfig
from tbp.monty.platform import Platform, PlatformConfig

__all__ = ["run"]


def run(platform_config: PlatformConfig, experiment_config: ExperimentConfig):
    platform = Platform(platform_config)

    try:
        platform.init()
        platform.run(experiment_config)
    finally:
        platform.shutdown()
