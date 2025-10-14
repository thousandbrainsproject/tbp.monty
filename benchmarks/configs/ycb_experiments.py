# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
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
    default_all_noise_params,
    default_all_noisy_sensor_module,
    default_evidence_1lm_config,
    default_evidence_lm_config,
    default_feature_weights,
    default_tolerance_values,
    default_tolerances,
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import YcbExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    CSVLoggingConfig,
    FiveLMMontySOTAConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewFartherAwaySOTAMontyConfig,
    PatchAndViewSOTAMontyConfig,
    SurfaceAndViewMontyConfig,
    SurfaceAndViewSOTAMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderMultiObjectArgs,
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import (
    DISTINCT_OBJECTS,
    SIMILAR_OBJECTS,
)
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.sensor_modules import (
    HabitatSM,
    Probe,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
    NoisySurfaceViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMultiObjectMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
(all use surface-agent models with 0.01 min dist and 64x64 resolution,
feature change SM, 50 min_steps, 500 max_steps)

To generate models run:
- supervised_pre_training_base
- only_surf_agent_training_10obj
- only_surf_agent_training_10simobj
- only_surf_agent_training_allobj
- supervised_pre_training_5lms
- supervised_pre_training_5lms_all_objects

QUICK EVALUATION TESTS:
- 10 objects, 32 known orientations, 1LM, distant agent
- 10 objects, 32 known orientations, 1LM, surface agent
- noise & 10 random rotations, distant agent
- noise & 10 random rotations, distant agent on model learned with distant sensor
- noise & 10 random rotations, surface agent
- no noise & 10 random rotations, surface agent
- noise & 10 random rotations, distant agent, 5LMs
- raw noise & 10 random rotations, surface agent
- 10 similar objects, 32 known orientations, surface agent
- 10 similar objects, noise & 10 random rotations, surface agent
- learning unsupervised

LONGER EVALUATION RUNS:
(all have 77 objects, best policy & paramters optimized for performance)
- default, 32 rotations, distant agent
- noise & 3 random rotations with 1LM, distant agent
- noise & 3 random rotations with 1LM, surface agent
- noise & 3 random rotations with 5LMs

For more details, see docs/how-to-use-monty/running-benchmarks.md
and docs/overview/benchmark-experiments.md
"""

# 14 unique rotations that give good views of the object. Same rotations used
# for supervised pretraining.
test_rotations_all = get_cube_face_and_corner_views_rotations()

# Limited number of rotations to use for quicker evaluation when doing longer
# runs with all 77 YCB objects.
test_rotations_3 = test_rotations_all[:3]

model_path_10distinctobj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_10distinctobj/pretrained/",
)

dist_agent_model_path_10distinctobj = os.path.join(
    pretrained_dir,
    "supervised_pre_training_base/pretrained/",
)

model_path_10simobj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_10similarobj/pretrained/",
)

model_path_5lms_10distinctobj = os.path.join(
    pretrained_dir,
    "supervised_pre_training_5lms/pretrained/",
)

model_path_1lm_77obj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_77obj/pretrained/",
)

model_path_5lms_77obj = os.path.join(
    pretrained_dir,
    "supervised_pre_training_5lms_all_objects/pretrained/",
)

# Default configs for surface policy which has a different desired object distance
default_surf_evidence_lm_config = copy.deepcopy(default_evidence_lm_config)
default_surf_evidence_lm_config["learning_module_args"]["gsg_args"][
    "desired_object_distance"
] = 0.025

# Using a smaller max_nneighbors for the noiseless experiments cuts the runtime
# in almost half without negatively affecting performance. For the noisy experiments a
# higher max_nneighbors is necessary so we use the default config above for
# those.
lower_max_nneighbors_lm_config = copy.deepcopy(default_evidence_lm_config)
lower_max_nneighbors_lm_config["learning_module_args"]["hypotheses_updater_args"][
    "max_nneighbors"
] = 5
lower_max_nneighbors_surf_lm_config = copy.deepcopy(default_surf_evidence_lm_config)
lower_max_nneighbors_surf_lm_config["learning_module_args"]["hypotheses_updater_args"][
    "max_nneighbors"
] = 5

lower_max_nneighbors_1lm_config = dict(learning_module_0=lower_max_nneighbors_lm_config)

default_evidence_surf_1lm_config = dict(
    learning_module_0=default_surf_evidence_lm_config
)
lower_max_nneighbors_surf_1lm_config = dict(
    learning_module_0=lower_max_nneighbors_surf_lm_config
)

lm0_config = copy.deepcopy(default_evidence_lm_config)
lm0_config["learning_module_args"]["tolerances"] = {"patch_0": default_tolerance_values}
lm1_config = copy.deepcopy(default_evidence_lm_config)
lm1_config["learning_module_args"]["tolerances"] = {"patch_1": default_tolerance_values}
lm2_config = copy.deepcopy(default_evidence_lm_config)
lm2_config["learning_module_args"]["tolerances"] = {"patch_2": default_tolerance_values}
lm3_config = copy.deepcopy(default_evidence_lm_config)
lm3_config["learning_module_args"]["tolerances"] = {"patch_3": default_tolerance_values}
lm4_config = copy.deepcopy(default_evidence_lm_config)
lm4_config["learning_module_args"]["tolerances"] = {"patch_4": default_tolerance_values}

default_5lm_lmconfig = dict(
    learning_module_0=lm0_config,
    learning_module_1=lm1_config,
    learning_module_2=lm2_config,
    learning_module_3=lm3_config,
    learning_module_4=lm4_config,
)

default_sensor_features_surf_agent = [
    "pose_vectors",
    "pose_fully_defined",
    "on_object",
    "object_coverage",
    "min_depth",
    "mean_depth",
    "hsv",
    "principal_curvatures",
    "principal_curvatures_log",
]

default_all_noisy_surf_agent_sensor_module = dict(
    sensor_module_class=HabitatSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        features=default_sensor_features_surf_agent,
        save_raw_obs=False,
        delta_thresholds={
            "on_object": 0,
            "distance": 0.01,
        },
        is_surface_sm=True,
        noise_params=default_all_noise_params,
    ),
)

sm0_config = copy.deepcopy(default_all_noisy_sensor_module)
sm0_config["sensor_module_args"]["sensor_module_id"] = "patch_0"

sm1_config = copy.deepcopy(default_all_noisy_sensor_module)
sm1_config["sensor_module_args"]["sensor_module_id"] = "patch_1"

sm2_config = copy.deepcopy(default_all_noisy_sensor_module)
sm2_config["sensor_module_args"]["sensor_module_id"] = "patch_2"

sm3_config = copy.deepcopy(default_all_noisy_sensor_module)
sm3_config["sensor_module_args"]["sensor_module_id"] = "patch_3"

sm4_config = copy.deepcopy(default_all_noisy_sensor_module)
sm4_config["sensor_module_args"]["sensor_module_id"] = "patch_4"

default_5sm_config = dict(
    sensor_module_0=sm0_config,
    sensor_module_1=sm1_config,
    sensor_module_2=sm2_config,
    sensor_module_3=sm3_config,
    sensor_module_4=sm4_config,
    sensor_module_5=dict(
        sensor_module_class=Probe,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=True,
        ),
    ),
)

base_config_10distinctobj_dist_agent = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=len(test_rotations_all),
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="benchmark_experiments",
        # Comment in for quick debugging (turns of wandb and increases logging)
        # wandb_handlers=[],
        # python_log_level="DEBUG",
    ),
    monty_config=PatchAndViewSOTAMontyConfig(
        learning_module_configs=lower_max_nneighbors_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

base_config_10distinctobj_surf_agent = copy.deepcopy(
    base_config_10distinctobj_dist_agent
)
base_config_10distinctobj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=len(test_rotations_all),
        max_total_steps=5000,  # x4 max_eval_steps for surface-policy, x2.5 for
        # feature-change SM
    ),
    monty_config=SurfaceAndViewSOTAMontyConfig(
        learning_module_configs=lower_max_nneighbors_surf_1lm_config,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
)

randrot_noise_10distinctobj_dist_agent = copy.deepcopy(
    base_config_10distinctobj_dist_agent
)
randrot_noise_10distinctobj_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=10,  # number of random rotations to test for each object
    ),
    monty_config=PatchAndViewSOTAMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

# NOTE: This experiment can reach a higher accuracy when using a 10% TH.
randrot_noise_10distinctobj_dist_on_distm = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
randrot_noise_10distinctobj_dist_on_distm.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=dist_agent_model_path_10distinctobj,
        n_eval_epochs=10,
    ),
)

randrot_noise_10distinctobj_surf_agent = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
randrot_noise_10distinctobj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=10,
        max_total_steps=5000,  # x4 max_eval_steps for surface-policy, x2.5 for
        # feature-change SM
    ),
    monty_config=SurfaceAndViewSOTAMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_surf_agent_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        learning_module_configs=default_evidence_surf_1lm_config,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
)

randrot_10distinctobj_surf_agent = copy.deepcopy(base_config_10distinctobj_surf_agent)
randrot_10distinctobj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=10,
        max_total_steps=5000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_10distinctobj_5lms_dist_agent = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
randrot_noise_10distinctobj_5lms_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms_10distinctobj,
        n_eval_epochs=10,
        min_lms_match=3,
    ),
    monty_config=FiveLMMontySOTAConfig(
        monty_class=MontyForEvidenceGraphMatching,  # has custom evidence voting method
        learning_module_configs=default_5lm_lmconfig,
        sensor_module_configs=default_5sm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
)

base_10simobj_surf_agent = copy.deepcopy(base_config_10distinctobj_surf_agent)
base_10simobj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10simobj,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=SIMILAR_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

randrot_noise_10simobj_surf_agent = copy.deepcopy(
    randrot_noise_10distinctobj_surf_agent
)
randrot_noise_10simobj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10simobj,
        n_eval_epochs=10,  # number of random rotations to test for each object
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=SIMILAR_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_10simobj_dist_agent = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
randrot_noise_10simobj_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_10simobj,
        n_eval_epochs=10,  # number of random rotations to test for each object
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=SIMILAR_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randomrot_rawnoise_10distinctobj_surf_agent = copy.deepcopy(
    randrot_noise_10distinctobj_surf_agent
)
randomrot_rawnoise_10distinctobj_surf_agent.update(
    dataset_args=NoisySurfaceViewFinderMountHabitatDatasetArgs(),
)

# Experiment with multiple (distractor objects)
base_10multi_distinctobj_dist_agent = copy.deepcopy(
    base_config_10distinctobj_dist_agent
)
base_10multi_distinctobj_dist_agent.update(
    # Agent is farther from object and takes larger steps to increase chance of
    # landing on other objects; uses hypothesis-testing policy (which may also cause
    # it to land on other objects)
    monty_config=PatchAndViewFartherAwaySOTAMontyConfig(
        learning_module_configs=lower_max_nneighbors_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    dataset_args=PatchViewFinderMultiObjectMountHabitatDatasetArgs(),
    eval_dataloader_args=EnvironmentDataloaderMultiObjectArgs(
        object_names=dict(
            targets_list=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
            source_object_list=DISTINCT_OBJECTS,
            num_distractors=10,  # Number of other objects added to the environment
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
)

# ----- Learning Unsupervised -----

default_lfs_lm = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances=default_tolerances,
            feature_weights=default_feature_weights,
            # Higher threshold to avoid too many false positives during the first
            # episodes. In the future we may consider some kind of learning schedule
            x_percent_threshold=50,
            graph_delta_thresholds=dict(
                patch=dict(
                    distance=0.01,
                    pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                    principal_curvatures_log=[1.0, 1.0],
                    hsv=[0.1, 1, 1],
                )
            ),
            # Since we have 100 min steps we want these steps to have been consistent
            # with the mlh. If we keep this value at its default (1) it often happens
            # that the episode just terminates once min steps is reached. Again, we
            # try to avoid too many false positive in the first epochs.
            object_evidence_threshold=100,
            # Also the symmetry evidence increments a lot after 100 steps and easily
            # reaches the default required evidence. Again, these are temporary fixes
            # and we will probably want some more stable long term solutions.
            required_symmetry_evidence=20,
            hypotheses_updater_args=dict(
                max_nneighbors=5,
            ),
        ),
    )
)

# NOTE: Using the MotorSystemConfigCurvatureInformedSurface does not work so well
# in this setting and currently leads to more confused objects.

surf_agent_unsupervised_10distinctobj = copy.deepcopy(
    base_config_10distinctobj_dist_agent
)
surf_agent_unsupervised_10distinctobj.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=10,
        max_train_steps=4000,
        max_total_steps=4000,
    ),
    logging_config=CSVLoggingConfig(python_log_level="INFO"),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=100),
        learning_module_configs=default_lfs_lm,
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

surf_agent_unsupervised_10distinctobj_noise = copy.deepcopy(
    surf_agent_unsupervised_10distinctobj
)
surf_agent_unsupervised_10distinctobj_noise.update(
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=100),
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_surf_agent_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        learning_module_configs=default_lfs_lm,
    )
)

surf_agent_unsupervised_10simobj = copy.deepcopy(surf_agent_unsupervised_10distinctobj)
surf_agent_unsupervised_10simobj.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=SIMILAR_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

# --------- Long Runs ----------------

base_77obj_dist_agent = copy.deepcopy(base_config_10distinctobj_dist_agent)
base_77obj_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_1lm_77obj,
        n_eval_epochs=len(test_rotations_3),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_3),
    ),
)

base_77obj_surf_agent = copy.deepcopy(base_config_10distinctobj_surf_agent)
base_77obj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_1lm_77obj,
        n_eval_epochs=len(test_rotations_3),
        max_total_steps=5000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_3),
    ),
)

randrot_noise_77obj_surf_agent = copy.deepcopy(randrot_noise_10distinctobj_surf_agent)
randrot_noise_77obj_surf_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_1lm_77obj,
        n_eval_epochs=3,
        max_total_steps=5000,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_77obj_dist_agent = copy.deepcopy(randrot_noise_10distinctobj_dist_agent)
randrot_noise_77obj_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_1lm_77obj,
        n_eval_epochs=3,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_77obj_5lms_dist_agent = copy.deepcopy(
    randrot_noise_10distinctobj_5lms_dist_agent
)
randrot_noise_77obj_5lms_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_5lms_77obj,
        n_eval_epochs=1,
        min_lms_match=3,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            77,
        ),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

experiments = YcbExperiments(
    base_config_10distinctobj_dist_agent=base_config_10distinctobj_dist_agent,
    base_config_10distinctobj_surf_agent=base_config_10distinctobj_surf_agent,
    randrot_noise_10distinctobj_dist_agent=randrot_noise_10distinctobj_dist_agent,
    randrot_noise_10distinctobj_dist_on_distm=randrot_noise_10distinctobj_dist_on_distm,
    randrot_noise_10distinctobj_surf_agent=randrot_noise_10distinctobj_surf_agent,
    randrot_10distinctobj_surf_agent=randrot_10distinctobj_surf_agent,
    randrot_noise_10distinctobj_5lms_dist_agent=randrot_noise_10distinctobj_5lms_dist_agent,
    base_10simobj_surf_agent=base_10simobj_surf_agent,
    randrot_noise_10simobj_surf_agent=randrot_noise_10simobj_surf_agent,
    randrot_noise_10simobj_dist_agent=randrot_noise_10simobj_dist_agent,
    randomrot_rawnoise_10distinctobj_surf_agent=randomrot_rawnoise_10distinctobj_surf_agent,
    base_10multi_distinctobj_dist_agent=base_10multi_distinctobj_dist_agent,
    # ------------- Not yet evaluated -------------
    surf_agent_unsupervised_10distinctobj=surf_agent_unsupervised_10distinctobj,
    surf_agent_unsupervised_10distinctobj_noise=surf_agent_unsupervised_10distinctobj_noise,
    surf_agent_unsupervised_10simobj=surf_agent_unsupervised_10simobj,
    # ------------- long runs with all objects -------------
    base_77obj_dist_agent=base_77obj_dist_agent,
    base_77obj_surf_agent=base_77obj_surf_agent,
    randrot_noise_77obj_surf_agent=randrot_noise_77obj_surf_agent,
    randrot_noise_77obj_dist_agent=randrot_noise_77obj_dist_agent,
    randrot_noise_77obj_5lms_dist_agent=randrot_noise_77obj_5lms_dist_agent,
)
CONFIGS = asdict(experiments)
