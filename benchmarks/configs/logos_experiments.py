# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os
from dataclasses import asdict

from benchmarks.configs.names import LogosExperiments
from benchmarks.configs.pretraining_experiments import supervised_pre_training_base
from tbp.monty.frameworks.config_utils.config_args import PatchAndViewMontyConfig
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments.logos_on_objs import LOGOS
from tbp.monty.frameworks.models.sensor_modules import Probe

# Import helper functions from my_experiments
from benchmarks.configs.my_experiments import (
    POSITIONS,
    make_2d_sensor_module_config,
    make_compositional_dataset_args,
    make_naive_scan_motor_config,
    make_object_dataloader_args,
)

# LOGOS specific constants
LOGO_ROTATIONS = [[0.0, 0.0, 0.0]]

####################
# LOGOS EXPERIMENTS #
####################
supervised_pretraining_logos = copy.deepcopy(supervised_pre_training_base)
supervised_pretraining_logos["logging_config"].run_name = "supervised_pretraining_logos"
supervised_pretraining_logos.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(POSITIONS) * len(LOGO_ROTATIONS),
    ),
    dataset_args=make_compositional_dataset_args(),
    train_dataloader_args=make_object_dataloader_args(LOGOS, POSITIONS, LOGO_ROTATIONS),
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=make_naive_scan_motor_config(step_size=1),
    ),
)

supervised_pretraining_logos_2d_sensor = copy.deepcopy(supervised_pretraining_logos)
supervised_pretraining_logos_2d_sensor[
    "logging_config"
].run_name = "supervised_pretraining_logos_2d_sensor"
# Update to use 2D sensor module
supervised_pretraining_logos_2d_sensor.update(
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=make_naive_scan_motor_config(step_size=1),
        sensor_module_configs=dict(
            sensor_module_0=make_2d_sensor_module_config(
                debug_save_dir=os.path.join(
                    os.path.expanduser("~"),
                    "tbp/feat.2d_sensor/results/debug_2d_edges",
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
    ),
)

experiments = LogosExperiments(
    supervised_pretraining_logos=supervised_pretraining_logos,
    supervised_pretraining_logos_2d_sensor=supervised_pretraining_logos_2d_sensor,
)
CONFIGS = asdict(experiments)
