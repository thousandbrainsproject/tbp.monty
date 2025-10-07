# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

from benchmarks.configs.names import MyExperiments

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
)
CONFIGS = asdict(experiments)
