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
    ALL_PART_OBJECTS,
    OBJECTS_WITH_LOGOS_LVL1,
    OBJECTS_WITH_LOGOS_LVL2,
    OBJECTS_WITH_LOGOS_LVL3,
    OBJECTS_WITH_LOGOS_LVL4,
)
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
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

model_path_monolithic_models_lvl1 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl1_monolithic_models/pretrained/",
)

model_path_part_models = os.path.join(
    pretrained_dir,
    "supervised_pre_training_curved_objects_after_flat_and_logo/pretrained/",
)

model_path_compositional_models_lvl1 = os.path.join(
    pretrained_dir,
    "supervised_pre_training_objects_with_logos_lvl1_comp_models/pretrained/",
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
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-testing goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.03,  # Distance from the object to the
                # agent that is considered "close enough" to the object
            ),
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,  # TODO: C - Scale with receptive field size
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
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-testing goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.03,  # Distance from the object to the
                # agent that is considered "close enough" to the object
            ),
        ),
    ),
)

# See level description in src/tbp/monty/frameworks/environments/logos_on_objs.py
infer_comp_base_config = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        n_eval_epochs=len(test_rotations_all),
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
    dataset_class=ED.EnvironmentDataset,
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
    ),
)

infer_comp_lvl1_with_monolithic_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl1_with_monolithic_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_monolithic_models_lvl1,
        n_eval_epochs=len(test_rotations_all),
    ),
)


infer_parts_with_part_models = copy.deepcopy(infer_comp_base_config)
infer_parts_with_part_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_part_models,
        n_eval_epochs=len(test_rotations_all),
        # show_sensor_output=True,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(ALL_PART_OBJECTS), object_list=ALL_PART_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)


infer_comp_lvl1_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl1_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl1,
        n_eval_epochs=len(test_rotations_all),
    ),
)

infer_comp_lvl2_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl2_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl2,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL2), object_list=OBJECTS_WITH_LOGOS_LVL2
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)


infer_comp_lvl3_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl3_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl3,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL3), object_list=OBJECTS_WITH_LOGOS_LVL3
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)

infer_comp_lvl4_with_comp_models = copy.deepcopy(infer_comp_base_config)
infer_comp_lvl4_with_comp_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models_lvl4,
        n_eval_epochs=len(test_rotations_all),
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL4), object_list=OBJECTS_WITH_LOGOS_LVL4
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)

experiments = CompositionalInferenceExperiments(
    infer_comp_lvl1_with_monolithic_models=infer_comp_lvl1_with_monolithic_models,
    infer_parts_with_part_models=infer_parts_with_part_models,
    infer_comp_lvl1_with_comp_models=infer_comp_lvl1_with_comp_models,
    infer_comp_lvl2_with_comp_models=infer_comp_lvl2_with_comp_models,
    infer_comp_lvl3_with_comp_models=infer_comp_lvl3_with_comp_models,
    infer_comp_lvl4_with_comp_models=infer_comp_lvl4_with_comp_models,
)
CONFIGS = asdict(experiments)
