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

import numpy as np

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.pretraining_experiments import supervised_pre_training_base
from tbp.monty.frameworks.config_utils.config_args import (
    PatchAndViewMontyConfig,
    MotorSystemConfigNaiveScanSpiral,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments.logos_on_objs import LOGOS, OBJECTS_WITH_LOGOS_LVL1
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDPoseSM
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMountHabitatDatasetArgs,
    EnvInitArgsPatchViewMount,
)
from tbp.monty.frameworks.run import print_config

train_rotations_all = get_cube_face_and_corner_views_rotations()

LOGO_POSITIONS = [[0.0, 1.5, 0.0], [-0.03, 1.5, 0.0], [0.03, 1.5, 0.0]]
LOGO_ROTATIONS = [[0.0, 0.0, 0.0]]

# Let's learn it using our usual sensor module first
# Config from feat.compositional_testbed
supervised_pretraining_logos = copy.deepcopy(supervised_pre_training_base)
supervised_pretraining_logos["logging_config"].run_name = "supervised_pretraining_logos"
supervised_pretraining_logos.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(LOGO_POSITIONS) * len(LOGO_ROTATIONS),
    ),
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsPatchViewMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, len(LOGOS), object_list=LOGOS),
        object_init_sampler=PredefinedObjectInitializer(
            positions=LOGO_POSITIONS,
            rotations=LOGO_ROTATIONS,
        ),
    ),
)
supervised_pretraining_logos.update(
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=1),
            ),
        ),
    ),
)

LVL1_POSITIONS = [[0.0, 1.5, 0.0], [-0.03, 1.5, 0.0], [0.03, 1.5, 0.0]]
# LVL1_ROTATIONS = [[0.0, 0.0, 0.0]]
LVL1_ROTATIONS = [[10, -15, 0]]

supervised_pretraining_lvl1 = copy.deepcopy(supervised_pre_training_base)
supervised_pretraining_lvl1["logging_config"].run_name = "supervised_pretraining_lvl1_step2"
supervised_pretraining_lvl1.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=len(LVL1_POSITIONS) * len(LVL1_ROTATIONS),
        do_eval=False,
    ),
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsPatchViewMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, len(OBJECTS_WITH_LOGOS_LVL1), object_list=OBJECTS_WITH_LOGOS_LVL1),
        object_init_sampler=PredefinedObjectInitializer(
            positions=LVL1_POSITIONS,
            rotations=LVL1_ROTATIONS,
        ),
    ),
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=1),
            ),
        ),
    ),
)


supervised_pretraining_logos_2d_sensor = copy.deepcopy(supervised_pretraining_logos)

# Update to use 2D sensor module
supervised_pretraining_logos_2d_sensor.update(
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=1),
            ),
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=TwoDPoseSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "edge_strength",
                        "coherence",
                        "pose_from_edge",
                    ],
                    save_raw_obs=True,
                    debug_visualize=True,
                    debug_save_dir=os.path.join(
                        os.path.expanduser("~"),
                        "tbp/feat.2d_sensor/results/debug_2d_edges",
                    ),
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
    ),
)
supervised_pretraining_logos_2d_sensor[
    "logging_config"
].run_name = "supervised_pretraining_logos_2d_sensor"

supervised_pretraining_lvl1_2d_sensor = copy.deepcopy(supervised_pretraining_lvl1)

supervised_pretraining_lvl1_2d_sensor.update(
    monty_config=PatchAndViewMontyConfig(
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=1),
            ),
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=TwoDPoseSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "edge_strength",
                        "coherence",
                        "pose_from_edge",
                    ],
                    save_raw_obs=True,
                    debug_visualize=True,
                    debug_save_dir=os.path.join(
                        os.path.expanduser("~"),
                        "tbp/feat.2d_sensor/results/debug_2d_edges_lvl1_oblique",
                    ),
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
    ),
)
supervised_pretraining_lvl1_2d_sensor[
    "logging_config"
].run_name = "supervised_pretraining_lvl1_oblique_2d_sensor"

UPSIDEDOWN_ROTATIONS = [[0, 0, 180]]
supervised_pretraining_lvl1_upsidedown_2d_sensor = copy.deepcopy(supervised_pretraining_lvl1_2d_sensor)
supervised_pretraining_lvl1_upsidedown_2d_sensor.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(LVL1_POSITIONS) * len(UPSIDEDOWN_ROTATIONS),
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, len(OBJECTS_WITH_LOGOS_LVL1), object_list=OBJECTS_WITH_LOGOS_LVL1),
        object_init_sampler=PredefinedObjectInitializer(
            positions=LVL1_POSITIONS,
            rotations=UPSIDEDOWN_ROTATIONS,
        ),
    ),
)
supervised_pretraining_lvl1_upsidedown_2d_sensor[
    "logging_config"
].run_name = "supervised_pretraining_lvl1_upsidedown_2d_sensor"

experiments = MyExperiments(
    supervised_pretraining_logos=supervised_pretraining_logos,
    supervised_pretraining_logos_2d_sensor=supervised_pretraining_logos_2d_sensor,
    supervised_pretraining_lvl1=supervised_pretraining_lvl1,
    supervised_pretraining_lvl1_2d_sensor=supervised_pretraining_lvl1_2d_sensor,
    supervised_pretraining_lvl1_upsidedown_2d_sensor=supervised_pretraining_lvl1_upsidedown_2d_sensor,
)
CONFIGS = asdict(experiments)
