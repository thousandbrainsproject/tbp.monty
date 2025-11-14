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

from benchmarks.configs.my_experiments import (
    fe_pretrain_dir,
    make_2d_sensor_module_config,
    make_compositional_dataset_args,
    make_naive_scan_motor_config,
    make_object_dataloader_args,
)
from benchmarks.configs.names import DiskExperiments
from benchmarks.configs.pretraining_experiments import supervised_pre_training_base
from benchmarks.configs.ycb_experiments import base_config_10distinctobj_dist_agent
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewMontyConfig,
    PatchAndViewSOTAMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EvalExperimentArgs,
    SupervisedPretrainingExperimentArgs,
)
from tbp.monty.frameworks.environments.logos_on_objs import OBJECTS_WITH_LOGOS_LVL1_DISK
from tbp.monty.frameworks.models.sensor_modules import Probe
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    ReproduceEpisodeHandler,
)

# Custom tolerances without principal_curvatures_log
CUSTOM_TOLERANCE_VALUES = {
    "hsv": np.array([0.1, 0.2, 0.2]),
}

CUSTOM_TOLERANCES = {"patch": CUSTOM_TOLERANCE_VALUES}

LEARN_POSITIONS = [
    [0.0, 1.5, 0.0],
    [-0.03, 1.5, 0.0],
    [0.03, 1.5, 0.0],
]


def generate_inference_positions(
    base_positions, n_samples_per_position=7, sigma=0.03, seed=42
):
    """Generate inference positions by adding Gaussian jitter to base.

    Args:
        base_positions: List of [x, y, z] base positions
        n_samples_per_position: Number of jittered samples per base position
        sigma: Standard deviation for Gaussian jitter
        seed: Random seed for reproducibility

    Returns:
        List of positions with Gaussian jitter applied
    """
    np.random.seed(seed)
    inference_positions = []
    for pos in base_positions:
        for _ in range(n_samples_per_position):
            jitter = np.random.normal(0, sigma, 3)
            inference_positions.append(
                [pos[0] + jitter[0], pos[1] + jitter[1], pos[2] + jitter[2]]
            )
    return inference_positions


INFERENCE_POSITIONS = generate_inference_positions(LEARN_POSITIONS)

# Model paths for pretrained models
model_path_disk_control = os.path.join(
    fe_pretrain_dir,
    "disk_learning_standard_control/pretrained/",
)

model_path_disk_2d = os.path.join(
    fe_pretrain_dir,
    "disk_learning_standard_2d_sensor/pretrained/",
)


def make_disk_learning_experiment_pair(run_name, rotations):
    """Create a pair of LVL1 experiments (control + 2D sensor) with rotations.

    Args:
        run_name: Name for the experiment run (used for logging and debug subdirs).
        rotations: List of rotation configurations.

    Returns:
        tuple: (control_config, 2d_sensor_config) experiment dictionaries.
    """
    # Create control experiment
    control_config = copy.deepcopy(supervised_pre_training_base)
    control_config.update(
        experiment_args=SupervisedPretrainingExperimentArgs(
            n_train_epochs=len(LEARN_POSITIONS) * len(rotations),
        ),
        logging_config=PretrainLoggingConfig(
            output_dir=fe_pretrain_dir,
            run_name=f"disk_learning_{run_name}_control",
        ),
        dataset_args=make_compositional_dataset_args(),
        train_dataloader_args=make_object_dataloader_args(
            OBJECTS_WITH_LOGOS_LVL1_DISK, LEARN_POSITIONS, rotations
        ),
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
        ),
    )

    # Create 2D sensor experiment
    sensor_2d_config = copy.deepcopy(control_config)
    sensor_2d_config.update(
        logging_config=PretrainLoggingConfig(
            output_dir=fe_pretrain_dir,
            run_name=f"disk_learning_{run_name}_2d_sensor",
        ),
        monty_config=PatchAndViewMontyConfig(
            motor_system_config=make_naive_scan_motor_config(step_size=1),
            sensor_module_configs=dict(
                sensor_module_0=make_2d_sensor_module_config(
                    debug_save_dir=os.path.join(
                        os.path.expanduser("~"),
                        f"tbp/feat.2d_sensor/results/{run_name}_2d_sensor",
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
    # Enable raw RGB saving for the 2D sensor module
    sensor_2d_config["monty_config"].sensor_module_configs[
        "sensor_module_0"
    ]["sensor_module_args"]["save_raw_rgb"] = True
    sensor_2d_config["monty_config"].sensor_module_configs[
        "sensor_module_0"
    ]["sensor_module_args"]["raw_rgb_base_dir"] = os.path.join(
        os.path.expanduser("~"),
        f"tbp/data/{run_name}_RGB",
    )

    return control_config, sensor_2d_config


#############################
# DISK LEARNING EXPERIMENTS #
#############################

disk_learning_control, disk_learning_2d = make_disk_learning_experiment_pair(
    "tbp_numenta_disks", [[0.0, 0.0, 0.0]]
)

##############################
# DISK INFERENCE EXPERIMENTS #
##############################


def make_disk_inference_experiment(model_path, use_2d_sensor=False, debug_subdir=None, debug_version=False):
    """Create inference experiment config for disk objects.

    Args:
        model_path: Path to pretrained model
        use_2d_sensor: Whether to use 2D sensor module
        debug_subdir: Optional debug subdirectory for 2D sensor

    Returns:
        dict: Inference experiment configuration
    """
    if debug_subdir is None and use_2d_sensor:
        debug_subdir = "debug_2d_inference_disk"

    # Start with base config from YCB experiments
    config = copy.deepcopy(base_config_10distinctobj_dist_agent)

    # Preserve learning module configs from base
    learning_module_configs = config["monty_config"].learning_module_configs

    # Override tolerances to remove principal_curvatures_log
    for lm_key in learning_module_configs:
        lm_config = learning_module_configs[lm_key]
        if "learning_module_args" in lm_config:
            lm_config["learning_module_args"]["tolerances"] = CUSTOM_TOLERANCES

    # Set save_raw_obs to True for all sensor modules if debugging
    if debug_version:
        sensor_module_configs = config["monty_config"].sensor_module_configs
        for sm_config in sensor_module_configs.values():
            sm_config["sensor_module_args"]["save_raw_obs"] = True
        num_eval_steps = 100
        logging_config = ParallelEvidenceLMLoggingConfig(
            wandb_group="disk_inference_experiments",
            monty_handlers=[
                BasicCSVStatsHandler,
                DetailedJSONHandler,
                ReproduceEpisodeHandler,
            ],
        )
    else:
        sensor_module_configs = config["monty_config"].sensor_module_configs
        num_eval_steps = 500
        logging_config = ParallelEvidenceLMLoggingConfig(
            wandb_group="disk_inference_experiments",
        )
    config.update(
        experiment_args=EvalExperimentArgs(
            model_name_or_path=model_path,
            n_eval_epochs=len(INFERENCE_POSITIONS),
            max_eval_steps=num_eval_steps,
        ),
        logging_config=logging_config,
        monty_config=PatchAndViewSOTAMontyConfig(
            learning_module_configs=learning_module_configs,
            sensor_module_configs=sensor_module_configs,
            motor_system_config=make_naive_scan_motor_config(step_size=5),
            monty_args=MontyArgs(min_eval_steps=50),
        ),
        dataset_args=make_compositional_dataset_args(),
        eval_dataloader_args=make_object_dataloader_args(
            OBJECTS_WITH_LOGOS_LVL1_DISK, INFERENCE_POSITIONS, [[0.0, 0.0, 0.0]]
        ),
    )

    # Add 2D sensor if requested
    if use_2d_sensor:
        config["monty_config"] = PatchAndViewSOTAMontyConfig(
            learning_module_configs=learning_module_configs,
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
            monty_args=MontyArgs(min_eval_steps=50),
        )

    return config


# Create four inference experiments: control/2d sensor on control/2d models
disk_inference_control_on_control = make_disk_inference_experiment(
    model_path=model_path_disk_control,
    use_2d_sensor=False,
)

disk_inference_control_on_2d = make_disk_inference_experiment(
    model_path=model_path_disk_2d,
    use_2d_sensor=False,
)

disk_inference_2d_on_control = make_disk_inference_experiment(
    model_path=model_path_disk_control,
    use_2d_sensor=True,
    debug_subdir="debug_2d_inference_disk_on_control",
)

disk_inference_2d_on_2d = make_disk_inference_experiment(
    model_path=model_path_disk_2d,
    use_2d_sensor=True,
    debug_subdir="debug_2d_inference_disk_on_2d",
)

# Make a 3D-on-3D debug experiment for visualization
debug_control_on_control = make_disk_inference_experiment(
    model_path=model_path_disk_control,
    use_2d_sensor=False,
    debug_version=True,
)

experiments = DiskExperiments(
    disk_learning_control=disk_learning_control,
    disk_learning_2d=disk_learning_2d,
    disk_inference_control_on_control=disk_inference_control_on_control,
    disk_inference_control_on_2d=disk_inference_control_on_2d,
    disk_inference_2d_on_control=disk_inference_2d_on_control,
    disk_inference_2d_on_2d=disk_inference_2d_on_2d,
    debug_control_on_control=debug_control_on_control,
)
CONFIGS = asdict(experiments)
