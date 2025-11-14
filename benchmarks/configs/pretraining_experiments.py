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
    CUBE_FACE_AND_CORNER_VIEW_ROTATIONS,
    FiveLMMontyConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
    EnvironmentInterfacePerObjectArgs,
    PredefinedObjectInitializer,
    SupervisedPretrainingExperimentArgs,
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
    HabitatSM,
    Probe,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatEnvInterfaceConfig,
    PatchViewFinderMountHabitatEnvInterfaceConfig,
    SurfaceViewFinderMontyWorldMountHabitatEnvInterfaceConfig,
    SurfaceViewFinderMountHabitatEnvInterfaceConfig,
)

# FOR SUPERVISED PRETRAINING: 14 unique rotations that give good views of the object.
train_rotations_all = CUBE_FACE_AND_CORNER_VIEW_ROTATIONS

monty_models_dir = os.getenv("MONTY_MODELS", "")

fe_pretrain_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v10")
)

pre_surf_agent_visual_training_model_path = os.path.join(
    fe_pretrain_dir, "supervised_pre_training_all_objects/pretrained/"
)

supervised_pre_training_base = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=SupervisedPretrainingExperimentArgs(
        n_train_epochs=len(train_rotations_all),
    ),
    logging=PretrainLoggingConfig(
        output_dir=fe_pretrain_dir,
        python_log_level="INFO",
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
        # use spiral policy for more even object coverage during learning
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),
    ),
    env_interface_config=PatchViewFinderMountHabitatEnvInterfaceConfig(),
    train_env_interface_class=ED.InformedEnvironmentInterface,
    train_env_interface_args=EnvironmentInterfacePerObjectArgs(
        object_names=DISTINCT_OBJECTS,
        object_init_sampler=PredefinedObjectInitializer(rotations=train_rotations_all),
    ),
)


experiments = PretrainingExperiments(
    supervised_pre_training_base=supervised_pre_training_base,
)
CONFIGS = asdict(experiments)
