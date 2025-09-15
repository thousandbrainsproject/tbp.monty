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
from benchmarks.configs.ycb_experiments import base_77obj_surf_agent
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


base_77obj_surf_agent_hyp1_rerun = copy.deepcopy(base_77obj_surf_agent)
base_77obj_surf_agent_hyp1_rerun["logging_config"].run_name = "base_77obj_surf_agent_hyp1_rerun"
base_77obj_surf_agent_hyp1_rerun["logging_config"] = CustomDetailedJSONLoggingConfig(
    episodes_to_save=[85,89,139,162,227],
)


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    base_77obj_surf_agent_hyp1_rerun=base_77obj_surf_agent_hyp1_rerun,
)
CONFIGS = asdict(experiments)
