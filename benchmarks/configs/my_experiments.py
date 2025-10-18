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

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.pretraining_experiments import supervised_pre_training_base
from tbp.monty.frameworks.config_utils.config_args import (
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments.logos_on_objs import (
    ANGLES,
    TBP_LOGOS,
    OBJECTS_WITH_LOGOS_LVL1,
)
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.sensor_modules import Probe
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDPoseSM
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)

train_rotations_all = get_cube_face_and_corner_views_rotations()

# Consolidated constants
POSITIONS = [
    [0.0, 1.5, 0.0],
    [-0.03, 1.5, 0.0],
    [0.03, 1.5, 0.0],
    [0.0, 1.53, 0.0],
    [-0.03, 1.53, 0.0],
    [0.03, 1.53, 0.0],
    [0.0, 1.47, 0.0],
    [-0.03, 1.47, 0.0],
    [0.03, 1.47, 0.0],
]
LVL1_ROTATIONS = [[10, -15, 0]]
UPSIDEDOWN_ROTATIONS = [[0, 0, 180]]


# Helper functions for common config patterns
def make_2d_sensor_module_config(debug_save_dir, sensor_id="patch"):
    """Create a 2D sensor module configuration with specified debug directory.

    Returns:
        dict: Sensor module configuration dictionary.
    """
    return dict(
        sensor_module_class=TwoDPoseSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_id,
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
            debug_save_dir=debug_save_dir,
        ),
    )


def make_naive_scan_motor_config(step_size=1):
    """Create a naive scan motor system configuration.

    Returns:
        MotorSystemConfigNaiveScanSpiral: Motor system configuration.
    """
    return MotorSystemConfigNaiveScanSpiral(
        motor_system_args=dict(
            policy_class=NaiveScanPolicy,
            policy_args=make_naive_scan_policy_config(step_size=step_size),
        ),
    )


def make_compositional_dataset_args():
    """Create dataset args for compositional objects.

    Returns:
        PatchViewFinderMountHabitatDatasetArgs: Dataset configuration.
    """
    return PatchViewFinderMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsPatchViewMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__
    )


def make_object_dataloader_args(object_list, positions, rotations):
    """Create dataloader args for objects with given positions and rotations.

    Returns:
        EnvironmentDataloaderPerObjectArgs: Dataloader configuration.
    """
    return EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(object_list), object_list=object_list
        ),
        object_init_sampler=PredefinedObjectInitializer(
            positions=positions,
            rotations=rotations,
        ),
    )


def make_lvl1_experiment_pair(rotation_name, rotations, debug_subdir=None):
    """Create a pair of LVL1 experiments (control + 2D sensor) with rotations.

    Args:
        rotation_name: Name for the rotation set (e.g., "standard", "oblique").
        rotations: List of rotation configurations.
        debug_subdir: Optional debug subdirectory name for 2D sensor.

    Returns:
        tuple: (control_config, 2d_sensor_config) experiment dictionaries.
    """
    if debug_subdir is None:
        debug_subdir = f"debug_2d_edges_lvl1_{rotation_name}"

    # Create control experiment
    control_config = copy.deepcopy(supervised_pre_training_base)
    control_config[
        "logging_config"
    ].run_name = f"supervised_pretraining_lvl1_{rotation_name}_control"
    control_config.update(
        experiment_args=ExperimentArgs(
            n_train_epochs=len(POSITIONS) * len(rotations),
            do_eval=False,
        ),
        dataset_args=make_compositional_dataset_args(),
        train_dataloader_args=make_object_dataloader_args(
            OBJECTS_WITH_LOGOS_LVL1, POSITIONS, rotations
        ),
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
        ),
    )

    # Create 2D sensor experiment
    sensor_2d_config = copy.deepcopy(control_config)
    sensor_2d_config[
        "logging_config"
    ].run_name = f"supervised_pretraining_lvl1_{rotation_name}_2d_sensor"
    sensor_2d_config.update(
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
            sensor_module_configs=dict(
                sensor_module_0=make_2d_sensor_module_config(
                    debug_save_dir=os.path.join(
                        os.path.expanduser("~"),
                        f"tbp/feat.2d_sensor/results/{debug_subdir}",
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

    return control_config, sensor_2d_config


########################################
# LVL1 ROTATION COMPARISON EXPERIMENTS #
########################################
# Standard rotation (0, 0, 0)
lvl1_standard_control, lvl1_standard_2d = make_lvl1_experiment_pair(
    "standard", [[0.0, 0.0, 0.0]]
)

# Oblique rotation
lvl1_oblique_control, lvl1_oblique_2d = make_lvl1_experiment_pair(
    "oblique", [[0, 0, 0], [10, -15, 0]]
)

# Upside down rotation
lvl1_upsidedown_control, lvl1_upsidedown_2d = make_lvl1_experiment_pair(
    "upsidedown", [[0, 0, 180]]
)


def make_angles_experiment_pair(rotation_name, rotations, debug_subdir=None):
    """Create a pair of ANGLES experiments (control + 2D sensor) with rotations.

    Args:
        rotation_name: Name for the rotation set (e.g., "standard").
        rotations: List of rotation configurations.
        debug_subdir: Optional debug subdirectory name for 2D sensor.

    Returns:
        tuple: (control_config, 2d_sensor_config) experiment dictionaries.
    """
    if debug_subdir is None:
        debug_subdir = f"debug_2d_edges_angles_{rotation_name}"

    # Create control experiment
    control_config = copy.deepcopy(supervised_pre_training_base)
    control_config[
        "logging_config"
    ].run_name = f"supervised_pretraining_angles_{rotation_name}_control"
    control_config.update(
        experiment_args=ExperimentArgs(
            n_train_epochs=len(POSITIONS) * len(rotations),
            do_eval=False,
        ),
        dataset_args=make_compositional_dataset_args(),
        train_dataloader_args=make_object_dataloader_args(ANGLES, POSITIONS, rotations),
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
        ),
    )

    # Create 2D sensor experiment
    sensor_2d_config = copy.deepcopy(control_config)
    sensor_2d_config[
        "logging_config"
    ].run_name = f"supervised_pretraining_angles_{rotation_name}_2d_sensor"
    sensor_2d_config.update(
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
            sensor_module_configs=dict(
                sensor_module_0=make_2d_sensor_module_config(
                    debug_save_dir=os.path.join(
                        os.path.expanduser("~"),
                        f"tbp/feat.2d_sensor/results/{debug_subdir}",
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

    return control_config, sensor_2d_config


####
# Angles
#
angles_standard_control, angles_standard_2d = make_angles_experiment_pair(
    "standard", [[0.0, 0.0, 0.0]], debug_subdir="30_angles_yb"
)


experiments = MyExperiments(
    supervised_pretraining_lvl1_standard_control=lvl1_standard_control,
    supervised_pretraining_lvl1_standard_2d_sensor=lvl1_standard_2d,
    supervised_pretraining_lvl1_oblique_control=lvl1_oblique_control,
    supervised_pretraining_lvl1_oblique_2d_sensor=lvl1_oblique_2d,
    supervised_pretraining_lvl1_upsidedown_control=lvl1_upsidedown_control,
    supervised_pretraining_lvl1_upsidedown_2d_sensor=lvl1_upsidedown_2d,
    supervised_pretraining_angles_standard_control=angles_standard_control,
    supervised_pretraining_angles_standard_2d_sensor=angles_standard_2d,
)
CONFIGS = asdict(experiments)
