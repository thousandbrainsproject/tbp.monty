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
from dataclasses import asdict

import numpy as np

from benchmarks.configs.defaults import (
    default_evidence_1lm_config,
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import MontyWorldExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
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
    EvalExperimentArgs,
    WorldImageDataloaderArgs,
    WorldImageDatasetArgs,
    WorldImageFromStreamDatasetArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment

"""
Experiments for a Monty model trained on photogrammetry-scanned objects, and
then evaluated either on these same objects in simulation, or on these objects in the
real world, using depth images from a mobile device.

To generate pretrained models, run only_surf_agent_training_numenta_lab_obj

NOTE these experiments do not currently support running with multi-processing,
therefore ensure you omit the -m flag when running these
"""

model_path_numenta_lab_obj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_numenta_lab_obj/pretrained/",
)

# Evaluation on real-world depth data, but trained on photogrammetry scanned objects
world_image_on_scanned_model = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_numenta_lab_obj,
        n_eval_epochs=1,
        max_eval_steps=500,
        show_sensor_output=False,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(wandb_group="benchmark_experiments"),
    monty_config=PatchAndViewMontyConfig(
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # Take larger steps (move 20 pixels at a time)
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    dataset_args=WorldImageDatasetArgs(
        env_init_args=EnvInitArgsMontyWorldStandardScenes()
    ),
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
    eval_dataloader_args={},
    logging_config=EvalEvidenceLMLoggingConfig(
        wandb_handlers=[], python_log_level="INFO"
    ),
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

experiments = MontyWorldExperiments(
    world_image_from_stream_on_scanned_model=world_image_from_stream_on_scanned_model,
    # ------------- Experiments for Benchmarks Table -------------
    world_image_on_scanned_model=world_image_on_scanned_model,
    dark_world_image_on_scanned_model=dark_world_image_on_scanned_model,
    bright_world_image_on_scanned_model=bright_world_image_on_scanned_model,
    hand_intrusion_world_image_on_scanned_model=hand_intrusion_world_image_on_scanned_model,
    multi_object_world_image_on_scanned_model=multi_object_world_image_on_scanned_model,
)
CONFIGS = asdict(experiments)
