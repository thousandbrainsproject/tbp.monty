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

from benchmarks.configs.names import CompositionalLearningExperiments
from benchmarks.configs.pretraining_experiments import supervised_pre_training_base
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigInformedGoalStateDriven,
    MotorSystemConfigNaiveScanSpiral,
    TwoLMStackedMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments.logos_on_objs import (
    CURVED_OBJECTS_WITHOUT_LOGOS,
    FLAT_OBJECTS_WITHOUT_LOGOS,
    LOGOS,
    OBJECTS_WITH_LOGOS_LVL1,
    OBJECTS_WITH_LOGOS_LVL2,
    OBJECTS_WITH_LOGOS_LVL3,
    OBJECTS_WITH_LOGOS_LVL4,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (
    ResamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.motor_policies import InformedPolicy, NaiveScanPolicy
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsTwoLMDistantStackedMount,
    TwoLMStackedDistantMountHabitatDatasetArgs,
)

# FOR SUPERVISED PRETRAINING: 14 unique rotations that give good views of the object.
train_rotations_all = get_cube_face_and_corner_views_rotations()

monty_models_dir = os.getenv("MONTY_MODELS", "")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v10")
)

two_stacked_constrained_lms_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.001,
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
            object_evidence_threshold=20,
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.001,  # TODO: C - Scale with receptive field size
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
        ),
    ),
)

two_stacked_constrained_lms_config_with_resampling = copy.deepcopy(
    two_stacked_constrained_lms_config
)
two_stacked_constrained_lms_config_with_resampling["learning_module_0"][
    "learning_module_args"
]["hypotheses_updater_class"] = ResamplingHypothesesUpdater
two_stacked_constrained_lms_config_with_resampling["learning_module_0"][
    "learning_module_args"
]["evidence_threshold_config"] = "all"
two_stacked_constrained_lms_config_with_resampling["learning_module_0"][
    "learning_module_args"
]["object_evidence_threshold"] = 1
two_stacked_constrained_lms_config_with_resampling["learning_module_0"][
    "learning_module_args"
]["gsg_class"] = EvidenceGoalStateGenerator
two_stacked_constrained_lms_config_with_resampling["learning_module_0"][
    "learning_module_args"
]["gsg_args"] = dict(
    goal_tolerances=dict(
        location=0.015,
    ),
    elapsed_steps_factor=10,
    min_post_goal_success_steps=20,
    x_percent_scale_factor=0.75,
    desired_object_distance=0.03,
    wait_growth_multiplier=1,  # Since learning, enable more frequent jumps
)

# ====== Learn Child / Part Objects ======

# For learning and evaluating compostional models with the logos-on-objects dataset,
# we first train on the flat objects without logos. These are viewed in multiple
# rotations, like the standard YCB objects.
supervised_pre_training_flat_objects_wo_logos = copy.deepcopy(
    supervised_pre_training_base
)
supervised_pre_training_flat_objects_wo_logos.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        learning_module_configs=two_stacked_constrained_lms_config,
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),  # use spiral policy for more even object coverage during learning
    ),
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsTwoLMDistantStackedMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__,
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(FLAT_OBJECTS_WITHOUT_LOGOS), object_list=FLAT_OBJECTS_WITHOUT_LOGOS
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)

# For learning the logos, we present them in a single rotation, but at multiple
# positions, as the naive scan policy otherwise samples peripheral points on the models
# poorly. This must be run after supervised_pre_training_flat_objects_wo_logos.
LOGO_POSITIONS = [[0.0, 1.5, 0.0], [-0.03, 1.5, 0.0], [0.03, 1.5, 0.0]]
LOGO_ROTATIONS = [[0.0, 0.0, 0.0]]

supervised_pre_training_logos_after_flat_objects = copy.deepcopy(
    supervised_pre_training_flat_objects_wo_logos
)
supervised_pre_training_logos_after_flat_objects.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(LOGO_POSITIONS) * len(LOGO_ROTATIONS),
        show_sensor_output=False,
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_flat_objects_wo_logos/pretrained/",
        ),
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
        learning_module_configs=two_stacked_constrained_lms_config,
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=1),
            )
        ),  # use spiral policy for more even object coverage during learning
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, len(LOGOS), object_list=LOGOS),
        object_init_sampler=PredefinedObjectInitializer(
            positions=LOGO_POSITIONS,
            rotations=LOGO_ROTATIONS,
        ),
    ),
)

# NOTE: we load the model trained on flat objects and logos, but we inheret from the
# config used for 3D "flat" objects, since it is similar in step-size, rotations, etc.
supervised_pre_training_curved_objects_after_flat_and_logo = copy.deepcopy(
    supervised_pre_training_flat_objects_wo_logos
)

supervised_pre_training_curved_objects_after_flat_and_logo.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_logos_after_flat_objects/pretrained/",
        ),
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0,
            len(CURVED_OBJECTS_WITHOUT_LOGOS),
            object_list=CURVED_OBJECTS_WITHOUT_LOGOS,
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)

# ====== Learning Compositional Objects ======

# Learn monolithic models on the compositional objects, i.e. where both the LLLM
# and the HLLM learn the compositional *objects*, but without a compitional *model.
# This must be run after supervised_pre_training_logos_after_flat_objects.
supervised_pre_training_objects_with_logos_lvl1_monolithic_models = copy.deepcopy(
    supervised_pre_training_flat_objects_wo_logos
)
supervised_pre_training_objects_with_logos_lvl1_monolithic_models.update(
    # We load the model trained on the individual objects
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_logos_after_flat_objects/pretrained/",
        ),
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL1), object_list=OBJECTS_WITH_LOGOS_LVL1
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)

supervised_pre_training_objects_with_logos_lvl1_comp_models = copy.deepcopy(
    supervised_pre_training_objects_with_logos_lvl1_monolithic_models
)

supervised_pre_training_objects_with_logos_lvl1_comp_models.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
        model_name_or_path=os.path.join(
            fe_pretrain_dir,
            "supervised_pre_training_logos_after_flat_objects/pretrained/",
        ),
        supervised_lm_ids=["learning_module_1"],
        min_lms_match=2,
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=100),
        learning_module_configs=two_stacked_constrained_lms_config,
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),  # use spiral policy for more even object coverage during learning
    ),
)

supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling = copy.deepcopy(
    supervised_pre_training_objects_with_logos_lvl1_comp_models
)

supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling.update(
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=0, min_train_steps=500),
        learning_module_configs=two_stacked_constrained_lms_config_with_resampling,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(
            motor_system_args=dict(
                policy_class=InformedPolicy,
                policy_args=make_informed_policy_config(
                    action_space_type="distant_agent_no_translation",
                    action_sampler_class=ConstantSampler,
                    rotation_degrees=2.0,  # As learning with random saccades + hypothesis jumps, make saccades smaller
                    use_goal_state_driven_actions=True,
                ),
            ),
        ),
    ),
)

MODEL_PATH_WITH_ALL_CHILD_OBJECTS = os.path.join(
    fe_pretrain_dir,
    "supervised_pre_training_curved_objects_after_flat_and_logo/pretrained/",
)

supervised_pre_training_objects_with_logos_lvl2_comp_models = copy.deepcopy(
    supervised_pre_training_objects_with_logos_lvl1_comp_models
)

supervised_pre_training_objects_with_logos_lvl2_comp_models.update(
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
        model_name_or_path=MODEL_PATH_WITH_ALL_CHILD_OBJECTS,
        supervised_lm_ids=["learning_module_1"],
        min_lms_match=2,
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL2), object_list=OBJECTS_WITH_LOGOS_LVL2
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)

supervised_pre_training_objects_with_logos_lvl3_comp_models = copy.deepcopy(
    supervised_pre_training_objects_with_logos_lvl2_comp_models
)

supervised_pre_training_objects_with_logos_lvl3_comp_models.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL3), object_list=OBJECTS_WITH_LOGOS_LVL3
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)


supervised_pre_training_objects_with_logos_lvl4_comp_models = copy.deepcopy(
    supervised_pre_training_objects_with_logos_lvl2_comp_models
)

supervised_pre_training_objects_with_logos_lvl4_comp_models.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECTS_WITH_LOGOS_LVL4), object_list=OBJECTS_WITH_LOGOS_LVL4
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=train_rotations_all,
        ),
    ),
)

experiments = CompositionalLearningExperiments(
    supervised_pre_training_flat_objects_wo_logos=supervised_pre_training_flat_objects_wo_logos,
    supervised_pre_training_logos_after_flat_objects=supervised_pre_training_logos_after_flat_objects,
    supervised_pre_training_curved_objects_after_flat_and_logo=supervised_pre_training_curved_objects_after_flat_and_logo,
    supervised_pre_training_objects_with_logos_lvl1_monolithic_models=supervised_pre_training_objects_with_logos_lvl1_monolithic_models,
    supervised_pre_training_objects_with_logos_lvl1_comp_models=supervised_pre_training_objects_with_logos_lvl1_comp_models,
    supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling=supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling,
    supervised_pre_training_objects_with_logos_lvl2_comp_models=supervised_pre_training_objects_with_logos_lvl2_comp_models,
    supervised_pre_training_objects_with_logos_lvl3_comp_models=supervised_pre_training_objects_with_logos_lvl3_comp_models,
    supervised_pre_training_objects_with_logos_lvl4_comp_models=supervised_pre_training_objects_with_logos_lvl4_comp_models,
)
CONFIGS = asdict(experiments)
