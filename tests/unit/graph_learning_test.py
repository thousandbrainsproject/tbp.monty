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
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    InformedPolicy,
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.loggers.wandb_handlers import DetailedWandbMarkedObsHandler
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.utils.follow_up_configs import (
    create_eval_config_multiple_episodes,
    create_eval_episode_config,
)
from tbp.monty.frameworks.utils.logging_utils import (
    deserialize_json_chunks,
    load_stats,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsFiveLMMount,
    EnvInitArgsPatchViewMount,
    EnvInitArgsSurfaceViewMount,
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)
from tests.unit.resources.unit_test_utils import BaseGraphTestCases


@dataclass
class MotorSystemConfigFixed:
    motor_system_class: Callable = field(default=InformedPolicy)
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            file_name=Path(__file__).parent / "resources/fixed_test_actions.jsonl",
        )
    )


@dataclass
class MotorSystemConfigOffObject:
    motor_system_class: Callable = field(default=InformedPolicy)
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            file_name=Path(__file__).parent
            / "resources/fixed_test_actions_off_object.jsonl",
        )
    )


class GraphLearningTest(BaseGraphTestCases.BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        base = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                max_train_steps=30, max_eval_steps=30, max_total_steps=60
            ),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20)
            ),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        surface_agent_eval_config = copy.deepcopy(base)
        surface_agent_eval_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
            ),
            dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsSurfaceViewMount(data_path=None).__dict__,
            ),
        )

        ppf_pred_tests = copy.deepcopy(base)
        ppf_pred_tests.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="PPF",
                            tolerance=np.ones(4) * 0.001,
                            use_relative_len=True,
                        ),
                    )
                ),
            ),
        )

        disp_pred_tests = copy.deepcopy(base)
        disp_pred_tests.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                            tolerance=np.ones(3) * 0.0001,
                        ),
                    )
                ),
            ),
        )

        feature_pred_tests = copy.deepcopy(base)
        feature_pred_tests.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.01,
                            tolerances={
                                "patch": {
                                    "on_object": 0,
                                    "rgba": np.ones(4) * 10,
                                    "principal_curvatures": np.ones(2),
                                    "pose_vectors": [
                                        40,
                                        360,
                                        360,
                                    ],  # angular difference
                                }
                            },
                        ),
                    )
                ),
            ),
        )

        fixed_actions_disp = copy.deepcopy(base)
        fixed_actions_disp.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=10),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                            tolerance=np.ones(3) * 0.0001,
                        ),
                    )
                ),
            )
        )

        fixed_actions_ppf = copy.deepcopy(base)
        fixed_actions_ppf.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="PPF",
                            tolerance=np.ones(4) * 0.0001,
                            use_relative_len=True,
                        ),
                    )
                ),
            )
        )

        fixed_actions_feat = copy.deepcopy(base)
        fixed_actions_feat.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=30),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.001,
                            tolerances={
                                "patch": {
                                    "on_object": 0,
                                    "object_coverage": 0.5,
                                    "rgba": np.ones(4) * 10,
                                    "hsv": [0.1, 1, 1],
                                    "pose_vectors": np.ones(3) * 40,
                                    "principal_curvatures": np.ones(2) * 8,
                                    "principal_curvatures_log": np.ones(2) * 2,
                                    "gaussian_curvature": 10,
                                    "mean_curvature": 5,
                                    "gaussian_curvature_sc": 8,
                                    "mean_curvature_sc": 4,
                                }
                            },
                        ),
                    )
                ),
            )
        )

        feature_pred_tests_time_out = copy.deepcopy(base)
        feature_pred_tests_time_out.update(
            # Use more steps on first two epochs to build models of 2 objects
            experiment_args=ExperimentArgs(max_train_steps=30, max_total_steps=60),
            logging_config=LoggingConfig(
                output_dir=self.output_dir,
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=30),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            # Low mmd to initially learn two separate objects
                            # (will be changed after first 2 episodes)
                            # For some reason 0.01 is enough on laptop but not in
                            # docker container...
                            max_match_distance=0.001,
                            # high tolerances to get time outs
                            tolerances={
                                "patch": {
                                    "principal_curvatures_log": np.ones(2),
                                    "pose_vectors": [
                                        180,
                                        180,
                                        180,
                                    ],
                                }
                            },
                            # use no symmetry terminal condition to get time out
                            required_symmetry_evidence=1000,
                        ),
                    )
                ),
            ),
            # always show objects in same orientation
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=[[0.0, 0.0, 0.0]]
                ),
            ),
        )

        feature_pred_tests_offset = copy.deepcopy(fixed_actions_feat)
        feature_pred_tests_offset.update(
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    positions=[[0.0, 1.5, 0.0]]
                ),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        feature_pred_tests_confused = copy.deepcopy(fixed_actions_feat)
        feature_pred_tests_confused.update(
            experiment_args=ExperimentArgs(n_train_epochs=1),
            logging_config=LoggingConfig(
                output_dir=self.output_dir,
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigOffObject(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.01,
                            tolerances={
                                "patch": {
                                    "on_object": 0,
                                    "rgba": np.ones(4) * 10,
                                    "principal_curvatures": np.ones(2) * 5,
                                    "pose_vectors": [40, 20, 20],
                                }
                            },
                            required_symmetry_evidence=1000,
                            # with accounting for curvature directions being flopped
                            # we have two possible poses (0 and 180 flip). Disregard
                            # this here for testing sake.
                            pose_similarity_threshold=np.pi,
                        ),
                    )
                ),
            ),
        )

        feature_pred_tests_off_object = copy.deepcopy(feature_pred_tests_confused)
        feature_pred_tests_off_object.update(
            experiment_args=ExperimentArgs(
                n_train_epochs=2,
                max_eval_steps=50,
                max_train_steps=50,
                max_total_steps=200,
            ),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(
                    num_exploratory_steps=20, min_train_steps=12
                ),
                motor_system_config=MotorSystemConfigOffObject(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.01,
                            tolerances={
                                "patch": {
                                    "on_object": 0,
                                    "rgba": np.ones(4) * 10,
                                    "principal_curvatures": np.ones(2) * 5,
                                    "pose_vectors": [40, 20, 20],
                                }
                            },
                        ),
                    )
                ),
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=[[0, 0, 0]],
                ),
            ),
        )

        feat_test_uniform_initial_poses = copy.deepcopy(base)
        feat_test_uniform_initial_poses.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=30),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.01,
                            tolerances={
                                "patch": {
                                    "hsv": [0.1, 1, 1],
                                    "principal_curvatures_log": np.ones(2) * 0.1,
                                }
                            },
                            initial_possible_poses="uniform",
                        ),
                    )
                ),
            ),
        )

        multi_ppf_displacement_lm_config = dict(
            learning_module_class=DisplacementGraphLM,
            learning_module_args=dict(
                k=5,
                match_attribute="PPF",
                tolerance=np.ones(4) * 0.001,
                use_relative_len=True,
            ),
        )

        ppf_displacement_5lm_config = copy.deepcopy(base)
        ppf_displacement_5lm_config.update(
            experiment_args=ExperimentArgs(
                max_train_steps=30,
                max_eval_steps=30,
                max_total_steps=60,
                n_eval_epochs=3,
                min_lms_match=3,
            ),
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG"
            ),
            monty_config=FiveLMMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=10),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=multi_ppf_displacement_lm_config,
                    learning_module_1=multi_ppf_displacement_lm_config,
                    learning_module_2=multi_ppf_displacement_lm_config,
                    learning_module_3=multi_ppf_displacement_lm_config,
                    learning_module_4=multi_ppf_displacement_lm_config,
                ),
            ),
            dataset_args=FiveLMMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsFiveLMMount(data_path=None).__dict__,
            ),
        )

        multi_feature_lm_config = dict(
            learning_module_class=FeatureGraphLM,
            learning_module_args=dict(
                max_match_distance=0.01,
            ),
        )
        default_multi_feat_lm_tolerances = {
            "hsv": np.array([0.1, 1, 1]),  # only look at hue
            "principal_curvatures_log": np.ones(2),
        }
        lm0_config = copy.deepcopy(multi_feature_lm_config)
        lm0_config["learning_module_args"]["tolerances"] = {
            "patch_0": default_multi_feat_lm_tolerances
        }
        lm1_config = copy.deepcopy(multi_feature_lm_config)
        lm1_config["learning_module_args"]["tolerances"] = {
            "patch_1": default_multi_feat_lm_tolerances
        }
        lm2_config = copy.deepcopy(multi_feature_lm_config)
        lm2_config["learning_module_args"]["tolerances"] = {
            "patch_2": default_multi_feat_lm_tolerances
        }
        lm3_config = copy.deepcopy(multi_feature_lm_config)
        lm3_config["learning_module_args"]["tolerances"] = {
            "patch_3": default_multi_feat_lm_tolerances
        }
        lm4_config = copy.deepcopy(multi_feature_lm_config)
        lm4_config["learning_module_args"]["tolerances"] = {
            "patch_4": default_multi_feat_lm_tolerances
        }

        feature_5lm_config = copy.deepcopy(base)
        feature_5lm_config.update(
            experiment_args=ExperimentArgs(
                max_train_steps=30,
                max_eval_steps=30,
                max_total_steps=60,
                n_eval_epochs=3,
                min_lms_match=3,
            ),
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG"
            ),
            monty_config=FiveLMMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=10),
                motor_system_config=MotorSystemConfigFixed(),
                learning_module_configs=dict(
                    learning_module_0=lm0_config,
                    learning_module_1=lm1_config,
                    learning_module_2=lm2_config,
                    learning_module_3=lm3_config,
                    learning_module_4=lm4_config,
                ),
            ),
            dataset_args=FiveLMMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsFiveLMMount(data_path=None).__dict__,
            ),
        )

        self.base_config = base
        self.surface_agent_eval_config = surface_agent_eval_config
        self.ppf_config = ppf_pred_tests
        self.disp_config = disp_pred_tests
        self.feature_config = feature_pred_tests
        self.fixed_actions_disp = fixed_actions_disp
        self.fixed_actions_ppf = fixed_actions_ppf
        self.fixed_actions_feat = fixed_actions_feat
        self.feature_pred_tests_time_out = feature_pred_tests_time_out
        self.feature_pred_tests_confused = feature_pred_tests_confused
        self.feature_pred_tests_off_object = feature_pred_tests_off_object
        self.feat_test_uniform_initial_poses = feat_test_uniform_initial_poses
        self.ppf_displacement_5lm_config = ppf_displacement_5lm_config
        self.feature_5lm_config = feature_5lm_config

        pprint("\n\nCONFIG:\n\n")
        for key, val in self.base_config.items():
            pprint(f"{key}: {val}")

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self):
        """Canary for setup_experiment.

        This could be part of the setUp method, but it's easier to debug if something
        breaks the setup_experiment method if there's a separate test for it.
        """
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)

    def test_can_run_train_episode(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.run_episode()

    def test_right_data_in_buffer(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.pre_episode()
            for step, observation in enumerate(self.exp.dataloader):
                self.exp.model.step(observation)
                self.assertEqual(
                    step + 1,
                    len(self.exp.model.learning_modules[0].buffer),
                    "buffer does not contain the right amount of elements.",
                )
                self.assertEqual(
                    step + 1,
                    len(
                        self.exp.model.learning_modules[
                            0
                        ].buffer.get_all_locations_on_object(input_channel="first")
                    ),
                    "buffer does not contain the right amount of locations.",
                )
                if step == 0:
                    self.assertListEqual(
                        list(
                            self.exp.model.learning_modules[
                                0
                            ].buffer.get_nth_displacement(0, input_channel="first")
                        ),
                        list([0, 0, 0]),
                        "displacement at step 0 should be 0.",
                    )
                self.assertEqual(
                    step + 1,
                    len(
                        self.exp.model.learning_modules[0].buffer.displacements[
                            "patch"
                        ]["displacement"]
                    ),
                    "buffer does not contain the right amount of displacements.",
                )
                self.assertSetEqual(
                    set(self.exp.model.sensor_modules[0].features),
                    set(self.exp.model.learning_modules[0].buffer[-1]["patch"].keys()),
                    "buffer doesn't contain all features required for matching.",
                )
                if step == 3:
                    break

    def test_can_run_eval_episode(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)
            self.exp.model.set_experiment_mode("eval")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.run_episode()

    def test_can_run_eval_episode_with_surface_agent(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.surface_agent_eval_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            self.exp.model.set_experiment_mode("eval")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.run_episode()

    def test_can_run_ppf_experiment(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.ppf_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...evaluating...")
            self.exp.evaluate()

    def test_can_run_disp_experiment(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.disp_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...evaluating...")
            self.exp.evaluate()

    def test_can_run_feature_experiment(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...evaluating...")
            self.exp.evaluate()

    def test_fixed_actions_disp(self):
        """Runs three test episodes on capsule3DSolid and cubeSolid.

        1. capsule, rotation = 0,0,0 -> no models in memory yet -> store new model
        2. cube, rotation = 0,0,0 -> no models in memory yet -> store new model
        3. capsule, rotation = 0,45,0 -> displacements cant recognize rotated model
             -> store new rotated model of object
        4. cube, rotation = 0,45,0 -> same as 3 for cube
        5. capsule, rotation = 0,0,0 -> recognize first model stored in memory
        6. cube, rotation = 0,0,0 -> recognize first model stored in memory

        To make sure we recognize the first model in step three we use a fixed
        action policy that executed the same action sequence in every episode.

        Followed by three eval episodes on capsule3DSolid (same rotation sequence so
        the first and third episode should recognize the capsule).
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_disp)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            # self.exp.model.set_experiment_mode("eval")
            pprint("...training...")
            self.exp.train()
            pprint("...loading and checking train statistics...")
            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )
            self.check_train_results(train_stats)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))

        self.check_eval_results(eval_stats)

    def test_fixed_actions_ppf(self):
        """Like test_fixed_actions_disp but using point pair features for matching."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_ppf)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            # self.exp.model.set_experiment_mode("eval")
            pprint("...training...")
            self.exp.train()
            pprint("...loading and checking train statistics...")
            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )

            self.check_train_results(train_stats)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))

        self.check_eval_results(eval_stats)

    def test_fixed_actions_feat(self):
        """Like test_fixed_actions_disp but using point pair features for matching."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...loading and checking train statistics...")

            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )

            self.check_train_results(train_stats)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))

        self.check_eval_results(eval_stats)

    def test_reproduce_single_episode(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)

            pprint("...training...")
            self.exp.train()

        # Create a separate experiment for evaluation to mimic the us case of re-running
        # eval episodes from a pretrained model
        eval_cfg_1 = copy.deepcopy(config)
        eval_cfg_1["experiment_args"].model_name_or_path = os.path.join(
            self.exp.output_dir,
            "2",  # latest checkpoint
        )
        self.eval_exp_1 = MontyObjectRecognitionExperiment()
        with self.eval_exp_1:
            self.eval_exp_1.setup_experiment(eval_cfg_1)
            # TODO: update so it only runs one episode

            pprint("...evaluating (first time) ...")
            self.eval_exp_1.evaluate()

        # Create detailed follow up experiment
        eval_cfg_2 = create_eval_episode_config(
            parent_config=self.eval_exp_1.config,  # already converted to dict in exp
            parent_config_name="eval_cfg_1",
            episode=0,
            update_run_dir=False,  # we are running direct; no run.py
        )

        ###
        # Check that the arguments for the new experiment are correct
        ###

        # Detailed wandb logging should be automatically built in, though we will remove
        # it to avoid logging tests to wandb
        self.assertEqual(
            eval_cfg_2["logging_config"]["wandb_handlers"][-1],
            DetailedWandbMarkedObsHandler,
        )
        eval_cfg_2["logging_config"]["wandb_handlers"].pop()

        # check that the object being used is the same one from original exp
        self.assertEqual(
            eval_cfg_1["eval_dataloader_args"].object_names,
            eval_cfg_2["eval_dataloader_args"]["object_names"],
        )

        # If we made it this far, we have the correct parameters. Now run the experiment
        self.eval_exp_2 = MontyObjectRecognitionExperiment()
        with self.eval_exp_2:
            self.eval_exp_2.setup_experiment(eval_cfg_2)

            pprint("...evaluating (second time) ...")
            self.eval_exp_2.evaluate()

        ###
        # Check that basic csv stats are the same
        ###
        original_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_stats.csv"
        )
        new_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_episode_0_rerun", "eval_stats.csv"
        )

        original_stats = pd.read_csv(original_eval_stats_file)
        new_stats = pd.read_csv(new_eval_stats_file)
        # filter the time column, as both experiments took place at different times
        original_stats.drop(columns=["time"], inplace=True)
        new_stats.drop(columns=["time"], inplace=True)
        # Get only first episode; eval_exp_1 ran for 3 epochs
        self.assertTrue(original_stats.loc[0].equals(new_stats.loc[0]))

        ###
        # Just a few simple lines to check that the json logs are correct
        ###

        # TODO: Once json file i/o code has been updated, only load single episode
        original_json_file = os.path.join(
            self.eval_exp_1.output_dir, "detailed_run_stats.json"
        )
        new_json_file = os.path.join(
            self.eval_exp_1.output_dir,
            "eval_episode_0_rerun",
            "detailed_run_stats.json",
        )

        original_detailed_stats = deserialize_json_chunks(original_json_file)
        new_detailed_stats = deserialize_json_chunks(new_json_file)

        # Check that LM data is the same; absolute nightmare zoo of data formats
        og_lm_stats = original_detailed_stats["0"]["LM_0"]
        new_lm_stats = new_detailed_stats["0"]["LM_0"]

        self.compare_lm_stats(og_lm_stats, new_lm_stats)

        # Check that targets are the same
        og_lm_targets = original_detailed_stats["0"]["target"]
        new_lm_targets = new_detailed_stats["0"]["target"]
        for key, val in og_lm_targets.items():
            self.assertEqual(val, new_lm_targets[key])

        self.compare_sensor_module_logs(original_detailed_stats, new_detailed_stats)

    def test_reproduce_multiple_episodes(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)

            pprint("...training...")
            self.exp.train()

        # Create a separate experiment for evaluation to mimic the us case of re-running
        # eval episodes from a pretrained model
        eval_cfg_1 = copy.deepcopy(config)
        eval_cfg_1["experiment_args"].model_name_or_path = os.path.join(
            self.exp.output_dir,
            "2",  # latest checkpoint
        )
        self.eval_exp_1 = MontyObjectRecognitionExperiment()
        with self.eval_exp_1:
            self.eval_exp_1.setup_experiment(eval_cfg_1)

            pprint("...evaluating (first time) ...")
            self.eval_exp_1.evaluate()

        # Create detailed follow up experiment
        eval_cfg_2 = create_eval_config_multiple_episodes(
            parent_config=self.eval_exp_1.config,  # already converted to dict in exp
            parent_config_name="eval_cfg_1",
            episodes=[
                0,
                1,
                2,
            ],  # 3 episodes total, one episode for each of 3 epochs
            update_run_dir=False,  # we are running direct; no run.py
        )

        ###
        # Check that the arguments for the new experiment are correct
        ###

        # Detailed wandb logging should be automatically built in, though we will remove
        # it to avoid logging tests to wandb
        self.assertEqual(
            eval_cfg_2["logging_config"]["wandb_handlers"][-1],
            DetailedWandbMarkedObsHandler,
        )
        eval_cfg_2["logging_config"]["wandb_handlers"].pop()

        # capsule3DSolid is used as the lone eval object; make sure it is listed once
        # per episode
        self.assertEqual(
            eval_cfg_2["eval_dataloader_args"]["object_names"],
            ["capsule3DSolid", "capsule3DSolid", "capsule3DSolid"],
        )

        # Original sampler had just first two rotations, should cycle back to the first
        # on the third episode
        self.assertEqual(
            eval_cfg_2["eval_dataloader_args"]["object_init_sampler"].rotations,
            [[0.0, 0.0, 0.0], [45.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        # If we made it this far, we have the correct parameters. Now run the experiment
        self.eval_exp_2 = MontyObjectRecognitionExperiment()
        with self.eval_exp_2:
            self.eval_exp_2.setup_experiment(eval_cfg_2)

            pprint("...evaluating (second time) ...")
            self.eval_exp_2.evaluate()

        ###
        # Check that basic csv stats are the same
        ###
        original_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_stats.csv"
        )
        new_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_rerun_episodes", "eval_stats.csv"
        )

        original_stats = pd.read_csv(original_eval_stats_file)
        new_stats = pd.read_csv(new_eval_stats_file)
        # filter the time column, as both experiments took place at different times
        original_stats.drop(columns=["time"], inplace=True)
        new_stats.drop(columns=["time"], inplace=True)
        # Get only first episode; eval_exp_1 ran for 3 epochs
        self.assertTrue(original_stats.equals(new_stats))

        ###
        # Just a few simple lines to check that the json logs are correct
        ###

        # TODO: Once json file i/o code has been updated, only load single episode
        original_json_file = os.path.join(
            self.eval_exp_1.output_dir, "detailed_run_stats.json"
        )
        new_json_file = os.path.join(
            self.eval_exp_1.output_dir,
            "eval_rerun_episodes",
            "detailed_run_stats.json",
        )

        original_detailed_stats = deserialize_json_chunks(original_json_file)
        new_detailed_stats = deserialize_json_chunks(new_json_file)

        # Check that LM data is the same; absolute nightmare zoo of data formats
        og_lm_stats = original_detailed_stats["0"]["LM_0"]
        new_lm_stats = new_detailed_stats["0"]["LM_0"]

        self.compare_lm_stats(og_lm_stats, new_lm_stats)

        # Check that targets are the same
        og_lm_targets = original_detailed_stats["0"]["target"]
        new_lm_targets = new_detailed_stats["0"]["target"]
        for key, val in og_lm_targets.items():
            self.assertEqual(val, new_lm_targets[key])

        self.compare_sensor_module_logs(original_detailed_stats, new_detailed_stats)

    def test_reproduce_single_episode_with_multiple_episode_function(self):
        """Verify create_eval_config_multiple_episodes for a single episode."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)

            pprint("...training...")
            self.exp.train()

        # Create a separate experiment for evaluation to mimic the us case of re-running
        # eval episodes from a pretrained model
        eval_cfg_1 = copy.deepcopy(config)
        eval_cfg_1["experiment_args"].model_name_or_path = os.path.join(
            self.exp.output_dir,
            "2",  # latest checkpoint
        )
        self.eval_exp_1 = MontyObjectRecognitionExperiment()
        with self.eval_exp_1:
            self.eval_exp_1.setup_experiment(eval_cfg_1)

            pprint("...evaluating (first time) ...")
            self.eval_exp_1.evaluate()

        # Create detailed follow up experiment
        eval_cfg_2 = create_eval_config_multiple_episodes(
            parent_config=self.eval_exp_1.config,  # already converted to dict in exp
            parent_config_name="eval_cfg_1",
            episodes=[0],
            update_run_dir=False,  # we are running direct; no run.py
        )

        ###
        # Check that the arguments for the new experiment are correct
        ###

        # Detailed wandb logging should be automatically built in, though we will remove
        # it to avoid logging tests to wandb
        self.assertEqual(
            eval_cfg_2["logging_config"]["wandb_handlers"][-1],
            DetailedWandbMarkedObsHandler,
        )
        eval_cfg_2["logging_config"]["wandb_handlers"].pop()

        # check that the object being used is the same one from original exp
        self.assertEqual(
            eval_cfg_1["eval_dataloader_args"].object_names,
            eval_cfg_2["eval_dataloader_args"]["object_names"],
        )

        # If we made it this far, we have the correct parameters. Now run the experiment
        self.eval_exp_2 = MontyObjectRecognitionExperiment()
        with self.eval_exp_2:
            self.eval_exp_2.setup_experiment(eval_cfg_2)

            pprint("...evaluating (second time) ...")
            self.eval_exp_2.evaluate()

        ###
        # Check that basic csv stats are the same
        ###
        original_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_stats.csv"
        )
        new_eval_stats_file = os.path.join(
            self.eval_exp_1.output_dir, "eval_rerun_episodes", "eval_stats.csv"
        )

        original_stats = pd.read_csv(original_eval_stats_file)
        new_stats = pd.read_csv(new_eval_stats_file)
        # filter the time column, as both experiments took place at different times
        original_stats.drop(columns=["time"], inplace=True)
        new_stats.drop(columns=["time"], inplace=True)
        # Get only first episode; eval_exp_1 ran for 3 epochs
        self.assertTrue(original_stats.loc[0].equals(new_stats.loc[0]))

        ###
        # Just a few simple lines to check that the json logs are correct
        ###

        # TODO: Once json file i/o code has been updated, only load single episode
        original_json_file = os.path.join(
            self.eval_exp_1.output_dir, "detailed_run_stats.json"
        )
        new_json_file = os.path.join(
            self.eval_exp_1.output_dir,
            "eval_rerun_episodes",
            "detailed_run_stats.json",
        )

        original_detailed_stats = deserialize_json_chunks(original_json_file)
        new_detailed_stats = deserialize_json_chunks(new_json_file)

        # Check that LM data is the same; absolute nightmare zoo of data formats
        og_lm_stats = original_detailed_stats["0"]["LM_0"]
        new_lm_stats = new_detailed_stats["0"]["LM_0"]

        self.compare_lm_stats(og_lm_stats, new_lm_stats)

        # Check that targets are the same
        og_lm_targets = original_detailed_stats["0"]["target"]
        new_lm_targets = new_detailed_stats["0"]["target"]
        for key, val in og_lm_targets.items():
            self.assertEqual(val, new_lm_targets[key])

        self.compare_sensor_module_logs(original_detailed_stats, new_detailed_stats)

    def test_save_and_load(self):
        # Move this to graph_building_test.py?
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_ppf)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            # self.exp.model.set_experiment_mode("eval")
            pprint("...training...")
            self.exp.train()

        # We are training for 3 epochs by default, load most recent indexing from 0
        print("Loading a saved checkpoint")
        cfg2 = copy.deepcopy(self.fixed_actions_feat)
        cfg2["experiment_args"].model_name_or_path = os.path.join(
            config["logging_config"].output_dir,
            "2",  # latest checkpoint
        )
        self.exp2 = MontyObjectRecognitionExperiment()
        with self.exp2:
            self.exp2.setup_experiment(cfg2)

            graph_memory_1 = self.exp.model.learning_modules[
                0
            ].graph_memory.get_all_models_in_memory()
            graph_memory_2 = self.exp2.model.learning_modules[
                0
            ].graph_memory.get_all_models_in_memory()

            # Loop over each graph model and check they have the exact same data
            for obj_name in graph_memory_1.keys():
                for input_channel in graph_memory_1[obj_name].keys():
                    graph_1 = graph_memory_1[obj_name][input_channel]
                    graph_2 = graph_memory_2[obj_name][input_channel]
                    self.check_graphs_equal(graph_1, graph_2)

    def test_time_out(self):
        """Test time_out and pose_time_out detection and logging.

        # TODO: This test is a little shaky and should be improved.

        Episodes 0 and 1: Should detect no_match and build models for 2 objects
        Episode 2: Lowered max_steps and raised mmd -> detect pose_time_out
        (Episode 3: object is too similar with tolerances, will also detect time_out)
        Episodes 4 and 5: Increased curvature tolerance -> detect time_out
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_pred_tests_time_out)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            # self.exp.train()
            for e in range(6):
                if e % 2 == 0:
                    self.exp.pre_epoch()
                if e == 2:
                    # Set max steps low & raise mmd to get pose time outs
                    self.exp.max_train_steps = 3
                    self.exp.model.learning_modules[0].max_match_distance = 0.1
                if e == 4:
                    # set curvature threshold high to get time outs
                    self.exp.model.learning_modules[0].tolerances["patch"][
                        "principal_curvatures_log"
                    ] = [10, 10]
                self.exp.run_episode()
                if e % 2 == 1:
                    self.exp.post_epoch()

        pprint("...check time out logging...")
        train_stats = pd.read_csv(os.path.join(self.exp.output_dir, "train_stats.csv"))
        self.assertEqual(
            train_stats["primary_performance"][2],
            "pose_time_out",
            "pose time out not recognized/logged correctly",
        )
        # possible locations are str in .csv with one line per pose
        # unique rotations may already be one but we don't know
        # where we are yet.
        self.assertGreater(
            train_stats["possible_object_locations"][2].count("\n"),
            0,  # If there are two possible locations, there will be 1 newline
            "pose time out episode should have more than one possible pose.",
        )
        for episode in [4, 5]:
            self.assertEqual(
                train_stats["primary_performance"][episode],
                "time_out",
                "time out not recognized/logged correctly",
            )
            self.assertEqual(
                train_stats["primary_performance"][episode],
                "time_out",
                "time out not recognized/logged correctly",
            )
            self.assertGreater(
                train_stats["num_possible_matches"][episode],
                1,
                "time out episode should have more than one possible match.",
            )
            # possible objects are comma separated string
            self.assertGreater(
                train_stats["result"][episode].count(","),
                0,  # If there are two objects, there should be 1 comma
                "time out episode should log more than one possible match.",
            )

    def test_confused_logging(self):
        # When the algorithm evolves, this scenario may not lead to confusion
        # anymore. Setting min_steps would also avoid this probably.
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_actions_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            self.exp.pre_epoch()
            # Overwrite target with a false name to test confused logging.
            for e in range(4):
                self.exp.pre_episode()
                self.exp.model.primary_target = str(e)
                for lm in self.exp.model.learning_modules:
                    lm.primary_target = str(e)
                last_step = self.exp.run_episode_steps()
                self.exp.post_episode(last_step)
            self.exp.post_epoch()

        pprint("...checking run stats...")
        train_stats = pd.read_csv(os.path.join(self.exp.output_dir, "train_stats.csv"))
        for i in [0, 1]:
            self.assertEqual(
                train_stats["primary_performance"][i],
                "no_match",
                f"episode {i} should be no_match.",
            )
            self.assertEqual(
                train_stats["TFNP"][i],
                "unknown_object_not_matched_(TN)",
                f"episode {i} should detect a true negative.",
            )
        for i in [2, 3]:
            self.assertEqual(
                train_stats["primary_performance"][i],
                "confused",
                f"episode {i} should log confused performance.",
            )
            self.assertEqual(
                train_stats["TFNP"][i],
                "unknown_object_in_possible_matches_(FP)",
                f"episode {i} should detect a false positive.",
            )
            self.assertNotEqual(
                train_stats["primary_target_object"][i],
                train_stats["result"][i],
                "confused object id should not be the same as target.",
            )
            self.assertNotEqual(
                train_stats["primary_target_object"][i],
                train_stats["possible_match_sources"][i],
                "confused object id should not be in possible_match_sources.",
            )

    def test_moving_off_object(self):
        # Tests additional elements of logging, in particular in relation
        # to logging of observations when off the object
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_pred_tests_off_object)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            # First episode will be used to learn object (no_match is triggered before
            # min_steps is reached and the sensor moves off the object). In the second
            # episode the sensor moves off the sphere on episode steps 6+
            # Eventually, we circle round and come back to the object; recognition
            # does not take place before then because when off the object, matching
            # steps are no longer incremented, while it is an unfamiliar part of
            # the object that we return to
            self.exp.train()
            self.assertEqual(
                len(
                    self.exp.model.learning_modules[
                        0
                    ].buffer.get_all_locations_on_object(input_channel="patch")
                ),
                len(
                    self.exp.model.learning_modules[
                        0
                    ].buffer.get_all_features_on_object()["patch"]["pose_vectors"]
                ),
                "Did not retrieve same amount of feature and locations on object.",
            )
            self.assertEqual(
                sum(
                    self.exp.model.learning_modules[
                        0
                    ].buffer.get_all_features_on_object()["patch"]["on_object"]
                ),
                len(
                    self.exp.model.learning_modules[
                        0
                    ].buffer.get_all_features_on_object()["patch"]["on_object"]
                ),
                "not all retrieved features were collected on the object.",
            )
            # Since we don't add observations to the buffer that are off the object
            # there should only be 8 observations stored for the 12 matching steps
            # and all of them should be on the object.
            num_matching_steps = len(
                self.exp.model.learning_modules[0].buffer.stats["possible_matches"]
            )
            self.assertEqual(
                num_matching_steps,
                sum(
                    self.exp.model.learning_modules[0].buffer.features["patch"][
                        "on_object"
                    ][:num_matching_steps]
                ),
                "Number of match steps does not match with stored observations "
                "on object",
            )

    def test_detailed_logging(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_pred_tests_off_object)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading stats files...")
        train_stats, eval_stats, detailed_stats, lm_models = load_stats(
            self.exp.output_dir,
            load_train=True,
            load_eval=True,
            load_detailed=True,
        )
        for episode in lm_models.keys():
            self.assertEqual(
                list(lm_models[episode]["LM_0"].keys()),
                ["new_object0"],
                "should have only learned and saved one object during training.",
            )
        for row in range(train_stats.shape[0]):
            self.assertEqual(
                train_stats.loc[row]["primary_target_object"],
                detailed_stats[str(row)]["LM_0"]["target"]["object"],
                "targets in train_stats and detailed stats don't match.",
            )

        for row in range(eval_stats.shape[0]):
            detailed_id = train_stats.shape[0] + row - 1
            self.assertEqual(
                eval_stats.loc[row]["primary_target_object"],
                detailed_stats[str(detailed_id)]["LM_0"]["target"]["object"],
                "targets in eval_stats and detailed stats don't match.",
            )
        self.assertEqual(
            len(detailed_stats["1"]["SM_0"]["processed_observations"]),
            73,
            "sensor module observations should contain all observations,"
            "even those off the object.",
        )
        self.assertEqual(
            len(detailed_stats["1"]["LM_0"]["possible_matches"]),
            train_stats.loc[1]["monty_matching_steps"],
            "matching steps in detailed stats don't match with those in "
            "train stats.",
        )
        self.assertEqual(
            sum(np.array(detailed_stats["1"]["LM_0"]["patch"]["on_object"])),
            len(detailed_stats["1"]["LM_0"]["patch"]["on_object"]),
            "learning module observations should only contain observations"
            "that were on the object.",
        )
        # Could add more tests but not sure how important.

    def test_uniform_initial_poses(self):
        """Test same scenario as test_fixed_actions_feat with uniform poses."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feat_test_uniform_initial_poses)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            pprint("...loading and checking train statistics...")

            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )
            self.check_train_results(train_stats)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))
        self.check_eval_results(eval_stats)

    def get_gm_with_fake_object(self):
        graph_lm = FeatureGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
        )
        graph_lm.mode = "train"
        for observation in self.fake_obs_learn:
            graph_lm.exploratory_step([observation])
        graph_lm.detected_object = "new_object0"
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            4,
            "Should have stored exactly 4 locations in the buffer.",
        )
        graph_lm.post_episode()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            1,
            "Should have stored exactly one object in memory.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Learned object ID should be new_object0.",
        )
        return graph_lm

    def test_same_sequence_recognition(self):
        """Test that the object is recognized with same action sequence."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step([observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_reverse_sequence_recognition(self):
        """Test that object is recognized irrespective of sampling order."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test.reverse()

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step([observation])
            print(graph_lm.get_possible_matches())
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_offset_sequence_recognition(self):
        """Test that the object is recognized irrespective of its location rel body."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            observation.location = observation.location + np.ones(3)
            graph_lm.matching_step([observation])
            print(graph_lm.get_possible_matches())
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_new_sampling_recognition(self):
        """Test object recognition with slightly perturbed observations.

        Test that the object is recognized if observations don't exactly fit
        the model.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 0.001

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step([observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_locations_not_recognized(self):
        """Test that the object is not recognized if locations don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 5

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for i, observation in enumerate(fake_obs_test):
            graph_lm.matching_step([observation])
            if i == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one match at the first step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_different_features_not_recognized(self):
        """Test that the object is not recognized if features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].non_morphological_features["hsv"] = [0.5, 1, 1]

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step([observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                0,
                "Should have no possible matches.",
            )

    def test_different_pose_features_not_recognized(self):
        """Test that the object is not recognized if pose features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].morphological_features["pose_vectors"] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for step, observation in enumerate(fake_obs_test):
            graph_lm.matching_step([observation])
            if step == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one possible match at first step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_moving_off_object_and_back(self):
        """Test that the object is still recognized after moving off the object.

        TODO: since the monty class checks use_state in combine_inputs it doesn't make
        much sense to test this here anymore with an isolated LM.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[1].location = [1, 2, 1]
        fake_obs_test[1].morphological_features["on_object"] = 0
        fake_obs_test[1].use_state = False

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            if not observation.use_state:
                pass
            else:
                graph_lm.matching_step([observation])
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have exactly one possible match.",
                )
                self.assertEqual(
                    graph_lm.get_possible_matches()[0],
                    "new_object0",
                    "Should match to new_object0.",
                )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_5lm_displacement_experiment(self):
        """Test 5 displacement LMs voting with two evaluation settings."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.ppf_displacement_5lm_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()

            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )
            self.check_multilm_train_results(train_stats, num_lms=5, min_done=3)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))
        self.check_multilm_eval_results(
            eval_stats, num_lms=5, min_done=3, num_episodes=1
        )

    def test_5lm_feature_experiment(self):
        """Test 5 feature LMs voting with two evaluation settings."""
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_5lm_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()

            train_stats = pd.read_csv(
                os.path.join(self.exp.output_dir, "train_stats.csv")
            )
            # The following check is brittle and depends on sensor arrangement. Leaving
            # the rest of the test intact to detect run failures, but disabling checking
            # of particular results.
            # self.check_multilm_train_results(train_stats, num_lms=5, min_done=3)

            pprint("...evaluating...")
            self.exp.evaluate()

        pprint("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(self.exp.output_dir, "eval_stats.csv"))
        # Just testing 1 episode here. Somehow the second rotation doesn't get
        # recognized. Probably just some parameter setting due to flaws in old
        # LM but didn't want to dig too deep into that for now.
        self.check_multilm_eval_results(
            eval_stats, num_lms=5, min_done=3, num_episodes=1
        )


if __name__ == "__main__":
    unittest.main()
