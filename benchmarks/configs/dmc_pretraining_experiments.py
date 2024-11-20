# ------------------------------------------------------------------------------
# Copyright (C) 2023 Numenta Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc. No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.
# ------------------------------------------------------------------------------
"""Supervised pretraining experiments.

This module defines a suite of supervised pretraining experiments. The core models
that experiments produce are:
 - `dist_agent_1lm`
 - `surf_agent_1lm`
 - `touch_agent_1lm`
 - `dist_agent_2lm`
 - `dist_agent_5lm`
 - `dist_agent_9lm`
 - `dist_agent_10lm`

 All of these models are trained on 77 YCB objects with 14 rotations each. The `touch`
model is the same as the `surf` model but without access to color information. 10
distinct object variants are also created for each of these models, so each of the
above also has a dual version with the suffix `_10distinctobj` (more on this below).

Most settings are the same as in the standard `pretraining_experiments.py` module.
The biggest exception here is the use of 14 unique rotations for training. This
improves training speed significantly and yields improved object models for distant
agents. For surface agents, the lower number of rotations entails less object
exploration overall (14 episodes vs 32), yielding slightly less complete object
models. To compensate for this, the learning modules used for surface and touch
agents models have more exploratory steps than distant agents (1000 vs 500) which
produces object models roughly equivalent to existing YCB v9 models.

On style: Unlike `pretraining experiments.py`, this module prefers functions to return
configs rather than copying and modifying them (for the most part). For example, we
have the functions `get_dist_patch_config()` and `get_surf_patch_config()` which return
default sensor module configs used for pretraining.

This approach has two main benefits:

 1. Make settings easier to find. Rather than following a chain of copied configs
    back to find which sensor or learning module an experiment uses, we can just look
    at the function that returns the config. In this way, the functions are an easy
    way to look up defaults.
 2. Parameterized configs. This is especially useful when creating multi-LM
    experiments or deleting color information from sensor or learning modules in the
    case of touch-only (no color) experiments.

The config 'getter'functions defined here are
 - `get_dist_displacement_lm_config`
 - `get_surf_displacement_lm_config`
 - `get_dist_patch_config`
 - `get_surf_patch_config`
 - `get_view_finder_config`
 - `get_dist_motor_config`
 - `get_surf_motor_config`

Names and logger args follow a specific pattern:
 - Model names follow the pattern `{SENSOR}_agent_{NUM_LMS}lm`, where `SENSOR` is one of
    `dist`, `surf`, or `touch`. The suffix `_10distinctobj` is added automatically for
    10 distinct object variants.
 - The experiment key is `pretrain_{MODEL_NAME}` (e.g., `pretrain_dist_agent_1lm`). By
    'experiment key', I mean the key used to identify the config in `CONFIGS`.
 - The logging config's `run_name` is the model name.
 - The logging config's `output_dir` is `PRETRAIN_DIR`.

This module also has a few conveniences that add to or modify configs.
 - A 10-distinct object variant is automatically generated for every config.
 - Eval dataloader arguments (required but unused) are added to each config.
 - Experiments are checked to make sure no two configs have the same `output_dir` /
   `run_name` pair to ensure there is no conflict in output paths.

These conveniences are implemented at the bottom of the module, after `CONFIGS` is
defined.
"""

import copy
import os
from pathlib import Path

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    NineLMMontyConfig,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    TenLMMontyConfig,
    TwoLMMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    FiveLMMountHabitatDatasetArgs,
    NineLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    SurfaceViewFinderMountHabitatDatasetArgs,
    TenLMMountHabitatDatasetArgs,
    TwoLMMountHabitatDatasetArgs,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS, SHUFFLED_YCB_OBJECTS
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)

# Specify default here

# - Experiment args
NUM_EXPLORATORY_STEPS_DIST = 500
NUM_EXPLORATORY_STEPS_SURF = 1000

# - Paths
monty_models_dir = os.getenv("MONTY_MODELS")
if not monty_models_dir:
    monty_models_dir = "~/tbp/results/monty/pretrained_models"
PRETRAIN_DIR = Path(monty_models_dir).expanduser() / "pretrained_ycb_dmc"

# - Define training rotations. Views from enclosing cube faces plus its corners.
ROTATIONS_14 = [
    np.array([0, 0, 0]),
    np.array([0, 90, 0]),
    np.array([0, 180, 0]),
    np.array([0, 270, 0]),
    np.array([90, 0, 0]),
    np.array([90, 180, 0]),
    np.array([35, 45, 0]),
    np.array([325, 45, 0]),
    np.array([35, 315, 0]),
    np.array([325, 315, 0]),
    np.array([35, 135, 0]),
    np.array([325, 135, 0]),
    np.array([35, 225, 0]),
    np.array([325, 225, 0]),
]


# ------------------------------------------------------------------------------
# Getter functions for learning modules, sensor modules, and motor configs.
# ------------------------------------------------------------------------------


def get_dist_displacement_lm_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get configuration for a displacement learning module.

    Convenience function that helps with sensor module IDs (particularly in
    a multi-sensor/LM configuration) and excluding color from graphs.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color-related features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the displacement learning module.

    """
    out = dict(
        learning_module_class=DisplacementGraphLM,
        learning_module_args=dict(
            k=5,
            match_attribute="displacement",
            tolerance=np.ones(3) * 0.0001,
            graph_delta_thresholds={
                sensor_module_id: dict(
                    distance=0.001,  # 1 mm 0.01 on ycb v9
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1, 1],
                    hsv=[0.1, 1, 1],
                )
            },
        ),
    )
    if not color:
        out["learning_module_args"]["graph_delta_thresholds"][sensor_module_id].pop(
            "hsv"
        )
    return out


def get_surf_displacement_lm_config(
    sensor_module_id: str = "patch",
    color: bool = True,
) -> dict:
    """Get configuration for a displacement learning module.

    Convenience function that helps with sensor module IDs (particularly in
    a multi-sensor/LM configuration) and excluding color from graphs.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color-related features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the displacement learning module.

    """
    out = dict(
        learning_module_class=DisplacementGraphLM,
        learning_module_args=dict(
            k=5,
            match_attribute="displacement",
            tolerance=np.ones(3) * 0.0001,
            graph_delta_thresholds={
                sensor_module_id: dict(
                    distance=0.01,
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1, 1],
                    hsv=[0.1, 1, 1],
                )
            },
        ),
    )
    if not color:
        out["learning_module_args"]["graph_delta_thresholds"][sensor_module_id].pop(
            "hsv"
        )
    return out


def get_dist_patch_config(sensor_module_id: str = "patch", color: bool = True) -> dict:
    """Get default distant patch config for pretraining.

    Provided as a convenience for handling sensor ID names and excluding
    color-related features from the config.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the surface patch sensor module.
    """
    out = dict(
        sensor_module_class=HabitatDistantPatchSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
                "pose_vectors",
                "pose_fully_defined",
                "on_object",
                "principal_curvatures",
                "principal_curvatures_log",
                "gaussian_curvature",
                "mean_curvature",
                "gaussian_curvature_sc",
                "mean_curvature_sc",
                "object_coverage",
                # non-morphological features (optional)
                "rgba",
                "hsv",
            ],
            save_raw_obs=True,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("rgba")
        out["sensor_module_args"]["features"].remove("hsv")
    return out


def get_surf_patch_config(sensor_module_id: str = "patch", color: bool = True) -> dict:
    """Get default surface patch config for pretraining.

    Provided as a convenience for handling sensor ID names and excluding
    color-related features from the config.

    Args:
        sensor_module_id (str): Identifier for the sensor module. Defaults to "patch".
        color (bool): Whether to include color features. Defaults to True.

    Returns:
        dict: Configuration dictionary for the surface patch sensor module.
    """
    out = dict(
        sensor_module_class=HabitatSurfacePatchSM,
        sensor_module_args=dict(
            sensor_module_id=sensor_module_id,
            features=[
                # morphological features (necessarry)
                "pose_vectors",
                "pose_fully_defined",
                "on_object",
                "object_coverage",
                "min_depth",
                "mean_depth",
                "principal_curvatures",
                "principal_curvatures_log",
                "gaussian_curvature",
                "mean_curvature",
                "gaussian_curvature_sc",
                "mean_curvature_sc",
                # non-morphological features (optional)
                "rgba",
                "hsv",
            ],
            save_raw_obs=True,
        ),
    )
    if not color:
        out["sensor_module_args"]["features"].remove("rgba")
        out["sensor_module_args"]["features"].remove("hsv")
    return out


def get_view_finder_config() -> dict:
    """Get default config for view finder.

    Returns:
        dict: Configuration dictionary for the view finder sensor module.

    """
    return dict(
        sensor_module_class=DetailedLoggingSM,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=True,
        ),
    )


def get_dist_motor_config(step_size: int = 5) -> MotorSystemConfigNaiveScanSpiral:
    """Get default distant motor config for pretraining.

    Returns:
        MotorSystemConfigNaiveScanSpiral: Configuration for the motor system for use
        with a distant agent.

    """
    return MotorSystemConfigNaiveScanSpiral(
        motor_system_args=make_naive_scan_policy_config(step_size=step_size)
    )


def get_surf_motor_config() -> MotorSystemConfigCurvatureInformedSurface:
    """Get default surface motor config for pretraining.

    Returns:
        MotorSystemConfigCurvatureInformedSurface: Configuration for the motor system
        for use with a surface agent.

    """
    return MotorSystemConfigCurvatureInformedSurface()


"""
Functions used for generating experiment variants.
--------------------------------------------------------------------------------
"""


def make_10distinctobj_variant(template: dict) -> dict:
    """Make 10 distinct object variants for a given config.

    Returns:
        dict: Copy of `template` config that trains on DISTINCT_OBJECTS dataset.
            The logging config's `run_name` is appended with "_10distinctobj".

    """
    config = copy.deepcopy(template)
    run_name = f"{config['logging_config'].run_name}_10distinctobj"
    config["logging_config"].run_name = run_name
    config["train_dataloader_args"].object_names = DISTINCT_OBJECTS
    return config


"""
1 LM models
--------------------------------------------------------------------------------
"""


pretrain_dist_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="dist_agent_1lm",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_displacement_lm_config(),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config(),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)


pretrain_surf_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="surf_agent_1lm",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=NUM_EXPLORATORY_STEPS_SURF
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_displacement_lm_config(),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_surf_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)


pretrain_touch_agent_1lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="touch_agent_1lm",
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(
            num_exploratory_steps=NUM_EXPLORATORY_STEPS_SURF
        ),
        learning_module_configs=dict(
            learning_module_0=get_surf_displacement_lm_config(color=False),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_surf_patch_config(color=False),
            sensor_module_1=get_view_finder_config(),
        ),
        motor_system_config=get_surf_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)

"""
2 LMs
--------------------------------------------------------------------------------
"""

pretrain_dist_agent_2lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="dist_agent_2lm",
    ),
    monty_config=TwoLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_displacement_lm_config("patch_0"),
            learning_module_1=get_dist_displacement_lm_config("patch_1"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config("patch_0"),
            sensor_module_1=get_dist_patch_config("patch_1"),
            sensor_module_2=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TwoLMMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)

"""
5 LMs
--------------------------------------------------------------------------------
"""


pretrain_dist_agent_5lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="dist_agent_5lm",
    ),
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_displacement_lm_config("patch_0"),
            learning_module_1=get_dist_displacement_lm_config("patch_1"),
            learning_module_2=get_dist_displacement_lm_config("patch_2"),
            learning_module_3=get_dist_displacement_lm_config("patch_3"),
            learning_module_4=get_dist_displacement_lm_config("patch_4"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config("patch_0"),
            sensor_module_1=get_dist_patch_config("patch_1"),
            sensor_module_2=get_dist_patch_config("patch_2"),
            sensor_module_3=get_dist_patch_config("patch_3"),
            sensor_module_4=get_dist_patch_config("patch_4"),
            sensor_module_5=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=FiveLMMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)

# %%
"""
9 LMs
--------------------------------------------------------------------------------
"""


pretrain_dist_agent_9lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="dist_agent_9lm",
    ),
    monty_config=NineLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_displacement_lm_config("patch_0"),
            learning_module_1=get_dist_displacement_lm_config("patch_1"),
            learning_module_2=get_dist_displacement_lm_config("patch_2"),
            learning_module_3=get_dist_displacement_lm_config("patch_3"),
            learning_module_4=get_dist_displacement_lm_config("patch_4"),
            learning_module_5=get_dist_displacement_lm_config("patch_5"),
            learning_module_6=get_dist_displacement_lm_config("patch_6"),
            learning_module_7=get_dist_displacement_lm_config("patch_7"),
            learning_module_8=get_dist_displacement_lm_config("patch_8"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config("patch_0"),
            sensor_module_1=get_dist_patch_config("patch_1"),
            sensor_module_2=get_dist_patch_config("patch_2"),
            sensor_module_3=get_dist_patch_config("patch_3"),
            sensor_module_4=get_dist_patch_config("patch_4"),
            sensor_module_5=get_dist_patch_config("patch_5"),
            sensor_module_6=get_dist_patch_config("patch_6"),
            sensor_module_7=get_dist_patch_config("patch_7"),
            sensor_module_8=get_dist_patch_config("patch_8"),
            sensor_module_9=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=NineLMMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)


"""
10 LMs
--------------------------------------------------------------------------------
"""


pretrain_dist_agent_10lm = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=len(ROTATIONS_14),
        do_eval=False,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=PRETRAIN_DIR,
        run_name="dist_agent_10lm",
    ),
    monty_config=TenLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=NUM_EXPLORATORY_STEPS_DIST),
        learning_module_configs=dict(
            learning_module_0=get_dist_displacement_lm_config("patch_0"),
            learning_module_1=get_dist_displacement_lm_config("patch_1"),
            learning_module_2=get_dist_displacement_lm_config("patch_2"),
            learning_module_3=get_dist_displacement_lm_config("patch_3"),
            learning_module_4=get_dist_displacement_lm_config("patch_4"),
            learning_module_5=get_dist_displacement_lm_config("patch_5"),
            learning_module_6=get_dist_displacement_lm_config("patch_6"),
            learning_module_7=get_dist_displacement_lm_config("patch_7"),
            learning_module_8=get_dist_displacement_lm_config("patch_8"),
            learning_module_9=get_dist_displacement_lm_config("patch_9"),
        ),
        sensor_module_configs=dict(
            sensor_module_0=get_dist_patch_config("patch_0"),
            sensor_module_1=get_dist_patch_config("patch_1"),
            sensor_module_2=get_dist_patch_config("patch_2"),
            sensor_module_3=get_dist_patch_config("patch_3"),
            sensor_module_4=get_dist_patch_config("patch_4"),
            sensor_module_5=get_dist_patch_config("patch_5"),
            sensor_module_6=get_dist_patch_config("patch_6"),
            sensor_module_7=get_dist_patch_config("patch_7"),
            sensor_module_8=get_dist_patch_config("patch_8"),
            sensor_module_9=get_dist_patch_config("patch_9"),
            sensor_module_10=get_view_finder_config(),
        ),
        motor_system_config=get_dist_motor_config(),
    ),
    # Set up environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TenLMMountHabitatDatasetArgs(),
    # Set up training dataloader.
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=SHUFFLED_YCB_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=ROTATIONS_14),
    ),
)


"""
Finalize configs
--------------------------------------------------------------------------------
"""
CONFIGS = {
    "pretrain_dist_agent_1lm": pretrain_dist_agent_1lm,
    "pretrain_surf_agent_1lm": pretrain_surf_agent_1lm,
    "pretrain_touch_agent_1lm": pretrain_touch_agent_1lm,
    "pretrain_dist_agent_2lm": pretrain_dist_agent_2lm,
    "pretrain_dist_agent_5lm": pretrain_dist_agent_5lm,
    "pretrain_dist_agent_9lm": pretrain_dist_agent_9lm,
    "pretrain_dist_agent_10lm": pretrain_dist_agent_10lm,
}

# Add 10 distinct object variants.
_new_configs = {}
for key, exp in CONFIGS.items():
    _config = make_10distinctobj_variant(exp)
    _new_configs[f"{key}_10distinctobj"] = _config
CONFIGS.update(_new_configs)
del _new_configs, _config


# Sanity check: make sure no two configs have the same output dir and run name.
_output_paths = []
for exp in CONFIGS.values():
    _output_dir = Path(exp["logging_config"].output_dir)
    _run_name = exp["logging_config"].run_name
    _run_name = _run_name if _run_name else key
    _path = _output_dir / _run_name
    assert _path not in _output_paths
    _output_paths.append(_path)
del _output_paths, _output_dir, _run_name, _path


# Peform final cleanup chores.
for _config in CONFIGS.values():
    # Make sure 'do_eval' is set to False.
    _config["experiment_args"].do_eval = False

    # Add unused (but required) eval dataloader configs.
    _config["eval_dataloader_class"] = ED.InformedEnvironmentDataLoader
    _config["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
        object_names=["mug"],
        object_init_sampler=PredefinedObjectInitializer(rotations=[[0, 0, 0]]),
    )
del _config
