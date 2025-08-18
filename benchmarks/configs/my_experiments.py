# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict
import copy

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.ycb_experiments import (
    randrot_noise_10distinctobj_dist_agent,
)
# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)
randrot_noise_10distinctobj_dist_agent_detailed = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
randrot_noise_10distinctobj_dist_agent_detailed["monty_config"].learning_module_configs[
    "learning_module_0"
]["learning_module_args"]["use_multithreading"] = False

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    randrot_noise_10distinctobj_dist_agent_detailed=randrot_noise_10distinctobj_dist_agent_detailed,
)
CONFIGS = asdict(experiments)
