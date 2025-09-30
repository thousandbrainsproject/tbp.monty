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

from benchmarks.configs.names import PretrainingExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
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
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.two_d_data import NUMENTA_OBJECTS
from tbp.monty.frameworks.environments.ycb import (
    DISTINCT_OBJECTS,
    SHUFFLED_YCB_OBJECTS,
    SIMILAR_OBJECTS,
)
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMontyWorldMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

# FOR SUPERVISED PRETRAINING: 14 unique rotations that give good views of the object.
train_rotations_all = get_cube_face_and_corner_views_rotations()

monty_models_dir = os.getenv("MONTY_MODELS")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v10")
)

pre_surf_agent_visual_training_model_path = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_all_objects/pretrained/"
)

supervised_pre_training_base = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.001,
                            # Only first pose vector (surface normal) is currently used
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1, 1],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
                # NOTE: Learning works with any LM type. For instance you can use
                # the following code to run learning with the EvidenceGraphLM:
                # learning_module_class=EvidenceGraphLM,
                # learning_module_args=dict(
                #     max_match_distance=0.01,
                #     tolerances={"patch": dict()},
                #     feature_weights=dict(),
                #     graph_delta_thresholds=dict(patch=dict(
                #         distance=0.001,
                #         pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                #         principal_curvatures_log=[1, 1],
                #         hsv=[0.1, 1, 1],
                #     )),
                # ),
                # NOTE: When learning with the EvidenceGraphLM or FeatureGraphLM, no
                # edges will be added to the learned graphs (also not needed for
                # matching) while learning with DisplacementGraphLM is a superset of
                # these, i.e. captures all necessary information to do inference with
                # any three of the LM types.
            )
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),  # use spiral policy for more even object coverage during learning
    ),
    dataset_args=PatchViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)


only_surf_agent_training_10obj = copy.deepcopy(supervised_pre_training_base)
only_surf_agent_training_10obj.update(
    experiment_args=ExperimentArgs(
        n_train_epochs=len(train_rotations_all),
        do_eval=False,
    ),
    monty_config=SurfaceAndViewMontyConfig(
        monty_args=MontyFeatureGraphArgs(num_exploratory_steps=1000),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.01,
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1.0, 1.0],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
            ),
        ),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSurfacePatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "min_depth",
                        "mean_depth",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
    ),
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        run_name="surf_agent_1lm_10distinctobj",
    ),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

only_surf_agent_training_10simobj = copy.deepcopy(only_surf_agent_training_10obj)
only_surf_agent_training_10simobj.update(
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        run_name="surf_agent_1lm_10similarobj",
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=SIMILAR_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

only_surf_agent_training_allobj = copy.deepcopy(only_surf_agent_training_10obj)
only_surf_agent_training_allobj.update(
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        run_name=f"surf_agent_1lm_{len(SHUFFLED_YCB_OBJECTS)}obj",
    ),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(SHUFFLED_YCB_OBJECTS), object_list=SHUFFLED_YCB_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

only_surf_agent_training_numenta_lab_obj = copy.deepcopy(only_surf_agent_training_10obj)
only_surf_agent_training_numenta_lab_obj.update(
    logging_config=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        run_name="surf_agent_1lm_numenta_lab_obj",
    ),
    dataset_args=SurfaceViewFinderMontyWorldMountHabitatDatasetArgs(),
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 12, object_list=NUMENTA_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

# TODO: these don't use the graph_delta_thresholds of the one LM experiments. Do we
# want to update that?
supervised_pre_training_5lms = copy.deepcopy(supervised_pre_training_base)
supervised_pre_training_5lms.update(
    monty_config=FiveLMMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),
    ),
    dataset_args=FiveLMMountHabitatDatasetArgs(),
)

supervised_pre_training_5lms_all_objects = copy.deepcopy(supervised_pre_training_5lms)
supervised_pre_training_5lms_all_objects.update(
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(SHUFFLED_YCB_OBJECTS), object_list=SHUFFLED_YCB_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)

experiments = PretrainingExperiments(
    supervised_pre_training_base=supervised_pre_training_base,
    supervised_pre_training_5lms=supervised_pre_training_5lms,
    supervised_pre_training_5lms_all_objects=supervised_pre_training_5lms_all_objects,
    only_surf_agent_training_10obj=only_surf_agent_training_10obj,
    only_surf_agent_training_10simobj=only_surf_agent_training_10simobj,
    only_surf_agent_training_allobj=only_surf_agent_training_allobj,
    only_surf_agent_training_numenta_lab_obj=only_surf_agent_training_numenta_lab_obj,
)
CONFIGS = asdict(experiments)
