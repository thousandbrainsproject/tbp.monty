# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigInformedNoTransStepS20,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvInitArgsMontyWorldBrightScenes,
    EnvInitArgsMontyWorldDarkScenes,
    EnvInitArgsMontyWorldHandIntrusionScenes,
    EnvInitArgsMontyWorldMultiObjectScenes,
    EnvInitArgsMontyWorldStandardScenes,
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PatchViewFinderMontyWorldMountHabitatDatasetArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    WorldImageDataloaderArgs,
    WorldImageDatasetArgs,
    WorldImageFromStreamDatasetArgs,
    get_env_dataloader_per_object_by_idx,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.two_d_data import NUMENTA_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
)

from .ycb_experiments import (
    default_all_noisy_sensor_module,
    default_evidence_1lm_config,
    fe_pretrain_dir,
    min_eval_steps,
)

"""
Experiments for a Monty model trained on photogrammetry-scanned objects, and
then evaluated either on these same objects in simulation, or on these objects in the
real world, using depth images from a mobile device.

To generate pretrained models, run only_surf_agent_training_numenta_lab_obj

NOTE these experiments do not currently support running with multi-processing,
therefore ensure you omit the -m flag when running these
"""

model_path_numenta_lab_obj = os.path.join(
    fe_pretrain_dir,
    "surf_agent_1lm_numenta_lab_obj/pretrained/",
)

test_rotations_one = [[0, 0, 0]]

# Base config for evlauating on the scanned objects in Habitat (i.e. simulation)
base_config_monty_world = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_numenta_lab_obj,
        n_eval_epochs=len(test_rotations_one),
        max_eval_steps=500,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(wandb_group="benchmark_experiments"),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # Take larger steps (move 20 pixels at a time)
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMontyWorldMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=12),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 12, object_list=NUMENTA_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_one),
    ),
)

# More challenging evaluation on photogrammetry objects; serves as a baseline
# for viewing these objects from a fixed, single view, but in simulation rather than
# with real-world data
# Note we therefore use the basic distant-agent policy without hypothesis-driven
# actions, to be more comparable to the constraints of inference on real-world data
randrot_noise_sim_on_scan_monty_world = copy.deepcopy(base_config_monty_world)
randrot_noise_sim_on_scan_monty_world.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_numenta_lab_obj,
        n_eval_epochs=10,
        max_eval_steps=500,
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 12, object_list=NUMENTA_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

# Evaluation on real-world depth data, but trained on photogrammetry scanned objects
world_image_on_scanned_model = copy.deepcopy(base_config_monty_world)
world_image_on_scanned_model.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_numenta_lab_obj,
        n_eval_epochs=1,
        max_eval_steps=500,
        show_sensor_output=False,
    ),
    dataset_args=WorldImageDatasetArgs(
        env_init_args=EnvInitArgsMontyWorldStandardScenes()
    ),
    train_dataloader_class=ED.SaccadeOnImageDataLoader,
    train_dataloader_args=WorldImageDataloaderArgs(),
    eval_dataloader_class=ED.SaccadeOnImageDataLoader,
    # TODO: write something akin to PredefinedObjectInitializer to automatically
    # determine these values
    eval_dataloader_args=WorldImageDataloaderArgs(
        scenes=list(np.repeat(range(12), 4)),
        versions=list(np.tile(range(4), 12)),
        # For debugging:
        # scenes=[0, 0, 0, 0],
        # versions=[0, 1, 2, 3],
    ),
)

# For live-demos, run inference where we constantly look for incoming data streamed
# from the mobile device
world_image_from_stream_on_scanned_model = copy.deepcopy(world_image_on_scanned_model)
world_image_from_stream_on_scanned_model.update(
    dataset_args=WorldImageFromStreamDatasetArgs(),
    eval_dataloader_class=ED.SaccadeOnImageFromStreamDataLoader,
    eval_dataloader_args=dict(),
)


bright_world_image_on_scanned_model = copy.deepcopy(world_image_on_scanned_model)
bright_world_image_on_scanned_model.update(
    dataset_args=WorldImageDatasetArgs(
        env_init_args=EnvInitArgsMontyWorldBrightScenes()
    ),
)

dark_world_image_on_scanned_model = copy.deepcopy(world_image_on_scanned_model)
dark_world_image_on_scanned_model.update(
    dataset_args=WorldImageDatasetArgs(env_init_args=EnvInitArgsMontyWorldDarkScenes()),
)

hand_intrusion_world_image_on_scanned_model = copy.deepcopy(
    world_image_on_scanned_model
)
hand_intrusion_world_image_on_scanned_model.update(
    dataset_args=WorldImageDatasetArgs(
        env_init_args=EnvInitArgsMontyWorldHandIntrusionScenes()
    ),
)

multi_object_world_image_on_scanned_model = copy.deepcopy(world_image_on_scanned_model)
multi_object_world_image_on_scanned_model.update(
    dataset_args=WorldImageDatasetArgs(
        env_init_args=EnvInitArgsMontyWorldMultiObjectScenes()
    ),
)

CONFIGS = dict(
    base_config_monty_world=base_config_monty_world,
    world_image_from_stream_on_scanned_model=world_image_from_stream_on_scanned_model,
    # ------------- Experiments for Benchmarks Table -------------
    randrot_noise_sim_on_scan_monty_world=randrot_noise_sim_on_scan_monty_world,
    world_image_on_scanned_model=world_image_on_scanned_model,
    dark_world_image_on_scanned_model=dark_world_image_on_scanned_model,
    bright_world_image_on_scanned_model=bright_world_image_on_scanned_model,
    hand_intrusion_world_image_on_scanned_model=hand_intrusion_world_image_on_scanned_model,
    multi_object_world_image_on_scanned_model=multi_object_world_image_on_scanned_model,
)
