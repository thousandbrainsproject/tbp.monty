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

from benchmarks.configs.defaults import (
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import CompositionalInferenceExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigInformedGoalStateDriven,
    ParallelEvidenceLMLoggingConfig,
    TwoLMStackedMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.logos_on_objs import (
    OBJECTS_WITH_LOGOS_LVL1,
    OBJECTS_WITH_LOGOS_LVL2,
    OBJECTS_WITH_LOGOS_LVL3,
    OBJECTS_WITH_LOGOS_LVL4,
    PARENT_TO_CHILD_MAPPING,
)
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (  # noqa: E501
    ResamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsTwoLMDistantStackedMount,
    TwoLMStackedDistantMountHabitatDatasetArgs,
)

# 14 unique rotations that give good views of the object. Same rotations used
# for supervised pretraining.
test_rotations_all = get_cube_face_and_corner_views_rotations()
# test_rotations_all = [[0.0, 0.0, 0.0]]
N_EVAL_EPOCHS = len(test_rotations_all)

# For an explanation of the different levels of difficulty, see logos_on_objs.py

model_path_monolithic_models_lvl1 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl1_monolithic_models/pretrained/",
)

model_path_compositional_models_lvl1 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl1_comp_models/pretrained/",
)

model_path_compositional_models_lvl1_resampling = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling/pretrained/",
)

model_path_compositional_models_lvl2 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl2_comp_models/pretrained/",
)

model_path_compositional_models_lvl3 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl3_comp_models/pretrained/",
)

model_path_compositional_models_lvl4 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl4_comp_models/pretrained/",
)


two_stacked_constrained_lms_inference_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_0": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            # Note graph-delta-thresholds are not used for grid-based models
            feature_weights={},
            max_graph_size=0.3,
            num_model_voxels_per_dim=200,
            max_nodes_per_graph=2000,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,
                ),
                elapsed_steps_factor=10,
                min_post_goal_success_steps=5,
                x_percent_scale_factor=0.75,
                desired_object_distance=0.03,
            ),
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_1": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                },
                # object Id currently is an int representation of the strings
                # in the object label so we keep this tolerance high. This is
                # just until we have added a way to encode object ID with some
                # real similarity measure.
                "learning_module_0": {"object_id": 1},
            },
            feature_weights={"learning_module_0": {"object_id": 1}},
            max_graph_size=0.4,
            num_model_voxels_per_dim=200,
            max_nodes_per_graph=2000,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,
                ),
                elapsed_steps_factor=10,
                min_post_goal_success_steps=5,
                x_percent_scale_factor=0.75,
                desired_object_distance=0.03,
            ),
        ),
    ),
)

two_stacked_constrained_resampling_lms_inf_config = copy.deepcopy(
    two_stacked_constrained_lms_inference_config
)
for lm_config in two_stacked_constrained_resampling_lms_inf_config.values():
    lm_args = lm_config["learning_module_args"]
    lm_args["hypotheses_updater_class"] = ResamplingHypothesesUpdater
    lm_args["evidence_threshold_config"] = "all"
    lm_args["object_evidence_threshold"] = 1

# See level description in src/tbp/monty/frameworks/environments/logos_on_objs.py
infer_comp_base_config = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        n_eval_epochs=N_EVAL_EPOCHS,
        min_lms_match=1,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="benchmark_experiments",
        # Comment in for quick debugging (turns of wandb and increases logging)
        # wandb_handlers=[],
        # python_log_level="INFO",
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        learning_module_configs=two_stacked_constrained_lms_inference_config,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsTwoLMDistantStackedMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__,
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL1), object_list=OBJECTS_WITH_LOGOS_LVL1
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
        parent_to_child_mapping=PARENT_TO_CHILD_MAPPING,
    ),
)

infer_comp_lvl1_with_monolithic_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl1_with_monolithic_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_monolithic_models_lvl1,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
)


infer_comp_lvl1_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl1_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl1,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
)

infer_comp_lvl1_with_comp_models_and_resampling = copy.deepcopy(
    infer_comp_lvl1_with_comp_models
)
infer_comp_lvl1_with_comp_models_and_resampling.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl1_resampling,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        learning_module_configs=two_stacked_constrained_resampling_lms_inf_config,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
)

infer_comp_lvl2_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl2_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl2,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL2), object_list=OBJECTS_WITH_LOGOS_LVL2
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
        parent_to_child_mapping=PARENT_TO_CHILD_MAPPING,
    ),
)


infer_comp_lvl3_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl3_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl3,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL3), object_list=OBJECTS_WITH_LOGOS_LVL3
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
        parent_to_child_mapping=PARENT_TO_CHILD_MAPPING,
    ),
)

infer_comp_lvl4_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl4_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl4,
        n_eval_epochs=N_EVAL_EPOCHS,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL4), object_list=OBJECTS_WITH_LOGOS_LVL4
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
        parent_to_child_mapping=PARENT_TO_CHILD_MAPPING,
    ),
)

experiments = CompositionalInferenceExperiments(
    infer_comp_lvl1_with_monolithic_models=infer_comp_lvl1_with_monolithic_models,
    infer_comp_lvl1_with_comp_models=infer_comp_lvl1_with_comp_models,
    infer_comp_lvl1_with_comp_models_and_resampling=infer_comp_lvl1_with_comp_models_and_resampling,
    infer_comp_lvl2_with_comp_models=infer_comp_lvl2_with_comp_models,
    infer_comp_lvl3_with_comp_models=infer_comp_lvl3_with_comp_models,
    infer_comp_lvl4_with_comp_models=infer_comp_lvl4_with_comp_models,
)
CONFIGS = asdict(experiments)
