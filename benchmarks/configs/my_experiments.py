# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
from dataclasses import asdict, dataclass, field
from typing import List

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.ycb_experiments import base_config_10distinctobj_surf_agent
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    ReproduceEpisodeHandler,
)


@dataclass
class CustomDetailedJSONLoggingConfig(DetailedEvidenceLMLoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            DetailedJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    save_per_episode: bool = True
    save_consolidated: bool = False
    episodes_to_save: List[int] = field(default_factory=list)


custom_logging = copy.deepcopy(base_config_10distinctobj_surf_agent)
custom_logging[
    "logging_config"
].run_name = "custom_logging"
custom_logging["logging_config"] = CustomDetailedJSONLoggingConfig(
    episodes_to_save=[1, 3, 5],
)

experiments = MyExperiments(
    custom_logging=custom_logging,
)
CONFIGS = asdict(experiments)