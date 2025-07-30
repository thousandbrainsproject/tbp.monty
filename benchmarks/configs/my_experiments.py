# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.ycb_experiments import (
    randrot_noise_10distinctobj_dist_agent,
)
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)

config = copy.deepcopy(randrot_noise_10distinctobj_dist_agent)
config["logging_config"] = DetailedEvidenceLMLoggingConfig(
    wandb_handlers=[],
)

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    randrot_noise_10distinctobj_dist_agent_detailed=config,
)
CONFIGS = asdict(experiments)
