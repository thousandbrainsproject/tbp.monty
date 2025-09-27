# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import shutil
import tempfile
import unittest
from pprint import pprint

import habitat_sim.utils as hab_utils
import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontySOTAConfig,
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigInformedGoalStateDriven,
    MotorSystemConfigInformedNoTransStepS20,
    MotorSystemConfigNaiveScanSpiral,
    MotorSystemConfigSurface,
    PatchAndViewMontyConfig,
    SurfaceAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderMultiObjectArgs,
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_curv_surface_policy_config,
    make_informed_policy_config,
    make_surface_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.motor_policies import (
    InformedPolicy,
    SurfacePolicy,
    SurfacePolicyCurvatureInformed,
    get_perc_on_obj_semantic,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict
from tbp.monty.frameworks.utils.transform_utils import numpy_to_scipy_quat
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsFiveLMMount,
    EnvInitArgsPatchViewFinderMultiObjectMount,
    EnvInitArgsPatchViewMount,
    EnvInitArgsSurfaceViewMount,
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMultiObjectMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()

        self.base_dist_agent_config = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                n_train_epochs=1,
                n_eval_epochs=1,
                max_train_steps=30,
                max_eval_steps=30,
                max_total_steps=60,
            ),
            logging_config=LoggingConfig(
                output_dir=self.output_dir,
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
            ),
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["cubeSolid", "capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.spiral_config = copy.deepcopy(self.base_dist_agent_config)
        self.spiral_config.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigNaiveScanSpiral(),
            ),
        )

        self.dist_agent_hypo_driven_config = copy.deepcopy(self.base_dist_agent_config)
        self.dist_agent_hypo_driven_config.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
            ),
        )

        self.base_surf_agent_config = copy.deepcopy(self.base_dist_agent_config)
        self.base_surf_agent_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigSurface(),
            ),
            dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsSurfaceViewMount(data_path=None).__dict__,
            ),
        )

        self.curv_informed_config = copy.deepcopy(self.base_surf_agent_config)
        self.curv_informed_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigCurvatureInformedSurface(),
            ),
        )

        self.surf_agent_hypo_driven_config = copy.deepcopy(self.base_surf_agent_config)
        self.surf_agent_hypo_driven_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
            ),
        )

        # ==== Setup more complex config for multi-LM experiments ====
        default_multi_lm_tolerances = {
            "hsv": np.array([0.1, 1, 1]),
            "principal_curvatures_log": np.ones(2) * 0.1,
        }

        multi_lm_config = dict(
            learning_module_class=EvidenceGraphLM,
            learning_module_args=dict(
                max_match_distance=0.0001,
                feature_weights={
                    "patch": {
                        "hsv": np.array([1, 0, 0]),
                    }
                },
            ),
        )

        # TODO H: automated/more convenient way to generate these configs
        lm0_config = copy.deepcopy(multi_lm_config)
        lm0_config["learning_module_args"]["tolerances"] = {
            "patch_0": default_multi_lm_tolerances
        }
        lm1_config = copy.deepcopy(multi_lm_config)
        lm1_config["learning_module_args"]["tolerances"] = {
            "patch_1": default_multi_lm_tolerances
        }
        lm2_config = copy.deepcopy(multi_lm_config)
        lm2_config["learning_module_args"]["tolerances"] = {
            "patch_2": default_multi_lm_tolerances
        }
        lm3_config = copy.deepcopy(multi_lm_config)
        lm3_config["learning_module_args"]["tolerances"] = {
            "patch_3": default_multi_lm_tolerances
        }
        lm4_config = copy.deepcopy(multi_lm_config)
        lm4_config["learning_module_args"]["tolerances"] = {
            "patch_4": default_multi_lm_tolerances
        }

        default_5lm_lmconfig = dict(
            learning_module_0=lm0_config,
            learning_module_1=lm1_config,
            learning_module_2=lm2_config,
            learning_module_3=lm3_config,
            learning_module_4=lm4_config,
        )

        self.dist_agent_hypo_driven_multi_lm_config = copy.deepcopy(
            self.dist_agent_hypo_driven_config
        )
        self.dist_agent_hypo_driven_multi_lm_config.update(
            experiment_args=ExperimentArgs(
                n_train_epochs=1,
                n_eval_epochs=1,
            ),
            monty_config=FiveLMMontySOTAConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                monty_class=MontyForEvidenceGraphMatching,
                learning_module_configs=default_5lm_lmconfig,
            ),
            dataset_args=FiveLMMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsFiveLMMount(data_path=None).__dict__,
            ),
        )

        # === Setup configs for adversarial settings like falling off the object ===

        # Config for distant agent that always moves in the same direction, so as
        # to fall off the object
        self.fixed_action_distant_config = copy.deepcopy(self.base_dist_agent_config)
        self.fixed_action_distant_config.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=5),
                motor_system_config=MotorSystemConfigInformedNoTransStepS20(
                    motor_system_args=dict(
                        policy_class=InformedPolicy,
                        policy_args=make_informed_policy_config(
                            action_space_type="distant_agent_no_translation",
                            action_sampler_class=ConstantSampler,
                            # Take large steps for a quick experiment
                            rotation_degrees=20.0,
                            use_goal_state_driven_actions=False,
                            switch_frequency=0,  # Overwrite default of 1.0
                        ),
                    ),
                ),
            ),
        )

        # As above, but of surface agent's tangential steps
        self.fixed_action_surface_config = copy.deepcopy(self.base_dist_agent_config)
        self.fixed_action_surface_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigSurface(
                    motor_system_args=dict(
                        policy_class=SurfacePolicy,
                        policy_args=make_surface_policy_config(
                            desired_object_distance=0.025,
                            alpha=0.0,  # Overwrite default of 0.1
                            use_goal_state_driven_actions=False,
                            translation_distance=0.05,
                        ),
                    ),
                ),
            ),
            dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsSurfaceViewMount(data_path=None).__dict__,
            ),
        )

        self.poor_initial_view_dist_agent_config = copy.deepcopy(
            self.base_dist_agent_config
        )
        self.poor_initial_view_dist_agent_config.update(
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    positions=[[0.0, 1.5, -0.2]]  # Object is farther away than typical
                ),
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigInformedNoTransStepS20(
                    motor_system_args=dict(
                        policy_class=InformedPolicy,
                        policy_args=make_informed_policy_config(
                            action_space_type="distant_agent_no_translation",
                            action_sampler_class=ConstantSampler,
                            rotation_degrees=5.0,
                            use_goal_state_driven_actions=False,
                            switch_frequency=1.0,
                            good_view_percentage=0.5,
                        ),
                    ),
                ),
            ),
        )

        self.poor_initial_view_surf_agent_config = copy.deepcopy(
            self.poor_initial_view_dist_agent_config
        )
        self.poor_initial_view_surf_agent_config.update(
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigSurface(
                    motor_system_args=dict(
                        policy_class=SurfacePolicy,
                        policy_args=make_surface_policy_config(
                            desired_object_distance=0.025,
                            alpha=0.0,
                            use_goal_state_driven_actions=False,
                            translation_distance=0.05,
                        ),
                    ),
                ),
            ),
            dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsSurfaceViewMount(data_path=None).__dict__,
            ),
        )

        self.poor_initial_view_multi_object_config = copy.deepcopy(
            self.base_dist_agent_config
        )
        self.poor_initial_view_multi_object_config.update(
            # For multi-objects, we test get good view at evaluation, because in
            # Monty we don't currently train with multiple objects in the environment
            dataset_args=PatchViewFinderMultiObjectMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewFinderMultiObjectMount(
                    data_path=None
                ).__dict__,
            ),
            eval_dataloader_args=EnvironmentDataloaderMultiObjectArgs(
                object_names=dict(
                    targets_list=["cubeSolid"],
                    source_object_list=["cubeSolid", "capsule3DSolid"],
                    num_distractors=10,
                ),
                object_init_sampler=PredefinedObjectInitializer(
                    positions=[[0.2, 1.5, -0.2]]  # Object is farther away *and* to
                    # the right
                ),
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigInformedNoTransStepS20(
                    motor_system_args=dict(
                        policy_class=InformedPolicy,
                        policy_args=make_informed_policy_config(
                            action_space_type="distant_agent_no_translation",
                            action_sampler_class=ConstantSampler,
                            rotation_degrees=5.0,
                            use_goal_state_driven_actions=False,
                            switch_frequency=1.0,
                            good_view_percentage=0.5,
                        ),
                    ),
                ),
            ),
        )

        self.rotated_cube_view_config = copy.deepcopy(
            self.poor_initial_view_dist_agent_config
        )
        self.rotated_cube_view_config.update(
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    positions=[[-0.1, 1.5, -0.2]],
                    rotations=[[45, 45, 45]],
                ),
            ),
            monty_config=SurfaceAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                motor_system_config=MotorSystemConfigSurface(
                    motor_system_args=dict(
                        policy_class=SurfacePolicy,
                        policy_args=make_surface_policy_config(
                            desired_object_distance=0.025,
                            alpha=0.0,
                            use_goal_state_driven_actions=False,
                            translation_distance=0.05,
                        ),
                    ),
                ),
            ),
            dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsSurfaceViewMount(data_path=None).__dict__,
            ),
        )

        # ==== Setup fake observations for testing principal-curvature policies ====
        fake_sender_id = "patch"
        default_obs_args = dict(
            location=np.array([0, 0, 0]),
            morphological_features={
                "pose_vectors": np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures": [-5, 10],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id=fake_sender_id,
            sender_type="SM",
        )
        fo_1 = copy.deepcopy(default_obs_args)
        fo_1["location"] = np.array([0.01, 0, 0])
        fo_2 = copy.deepcopy(default_obs_args)
        fo_2["location"] = np.array([0.02, 0, 0])
        fo_3 = copy.deepcopy(default_obs_args)
        fo_3["location"] = np.array([0.02, 0.01, 0])

        # No well defined PC directions
        fo_4 = copy.deepcopy(default_obs_args)
        fo_4["location"] = np.array([0.02, 0.02, 0])
        fo_4["morphological_features"]["pose_fully_defined"] = False

        fo_5 = copy.deepcopy(default_obs_args)
        fo_5["location"] = np.array([0.03, 0.03, 0])

        self.fake_obs_pc = [
            State(**default_obs_args),
            State(**fo_1),
            State(**fo_2),
            State(**fo_3),
            State(**fo_4),
            State(**fo_5),
        ]

        # PC direciton "flipped", pointing back to a location we've already been at
        fo_1_backtrack_pc = copy.deepcopy(fo_1)
        fo_1_backtrack_pc["morphological_features"]["pose_vectors"] = np.array(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        )

        # PC direction is defined in z direction; here, as the sensor/agent is not
        # rotated, this means it is pointing towards/away from the sensor, rather than
        # orthogonal to it; in experiments, PC vectors pointing towards +z in the
        # reference frame of the sensor/agent can happen if the surface agent has failed
        # to orient such that it is looking down at the surface normal
        fo_2_corrupt_z = copy.deepcopy(fo_2)
        fo_2_corrupt_z["morphological_features"]["pose_vectors"] = np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        )

        self.fake_obs_advanced_pc = [
            State(**default_obs_args),
            State(**fo_1_backtrack_pc),
            State(**fo_2_corrupt_z),
        ]

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    # ==== BASIC UNIT TESTS FOR LOADING VARIOUS ACTION POLICIES ====

    # @unittest.skip("debugging")
    def test_can_run_informed_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.base_dist_agent_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_spiral_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.spiral_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            # TODO: test that no two locations are the same
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_dist_agent_hypo_driven_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.dist_agent_hypo_driven_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_surface_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.base_surf_agent_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_curv_informed_policy(self) -> None:
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.curv_informed_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_surf_agent_hypo_driven_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.surf_agent_hypo_driven_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # @unittest.skip("debugging")
    def test_can_run_multi_lm_dist_agent_hypo_driven_policy(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.dist_agent_hypo_driven_multi_lm_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            pprint("...evaluating...")
            exp.evaluate()

    # ==== MORE INVOLVED TESTS OF ACTION POLICIES ====

    def initialize_motor_system(self, config_object):
        """Setups up a motor system for testing.

        Returns:
            motor_system: created motor system instance
            motor_system_args: motor system arguments for reference
        """
        motor_system_config = config_to_dict(config_object)
        motor_system_class = motor_system_config["motor_system_class"]
        motor_system_args = motor_system_config["motor_system_args"]
        policy_class = motor_system_args["policy_class"]
        policy_args = motor_system_args["policy_args"]
        policy = policy_class(rng=np.random.RandomState(123), **policy_args)
        motor_system = motor_system_class(policy=policy)
        motor_system.pre_episode()

        return motor_system, motor_system_args

    def initialize_lm_with_gsg(self):
        """Setups up an LM with a goal-state generator for testing.

        Returns:
            graph_lm: Created evidence graph LM instance
            gsg_args: Goal-state generator arguments for reference
        """
        gsg_args = dict(
            elapsed_steps_factor=10,
            min_post_goal_success_steps=5,
            x_percent_scale_factor=0.75,
            desired_object_distance=0.025,
        )

        graph_lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=gsg_args,
        )
        return graph_lm, gsg_args

    def test_get_good_view_basic_dist_agent(self):
        """Test ability to move a distant agent to a good view of an object.

        Given a substandard view of an object, the "experimenter" (via agent actions)
        can move a distant agent to a good view of the object before beginning the
        experiment.

        In this basic version, the object is a bit too far away, and so the agent
        moves forward
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.poor_initial_view_dist_agent_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.pre_episode()

            pprint("...stepping through observations...")

            # Check the initial view
            observation = next(exp.dataloader)
            # TODO M remove the following train-wreck during refactor
            view = observation[exp.model.motor_system._policy.agent_id]["view_finder"]
            semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
            perc_on_target_obj = get_perc_on_obj_semantic(semantic, semantic_id=1)

            dict_config = config_to_dict(config)

            target_perc_on_target_obj = dict_config["monty_config"][
                "motor_system_config"
            ]["motor_system_args"]["policy_args"]["good_view_percentage"]

            assert perc_on_target_obj >= target_perc_on_target_obj, (
                f"Initial view is not good enough, {perc_on_target_obj} "
                f"vs target of {target_perc_on_target_obj}"
            )

            points_on_target_obj = semantic == 1
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            target_closest_point = dict_config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["desired_object_distance"]

            # Utility policy should not have moved too close to the object
            assert closest_point_on_target_obj > target_closest_point, (
                f"Initial view is too close, {closest_point_on_target_obj} "
                f"vs target of {target_closest_point}"
            )

    def test_touch_object_basic_surf_agent(self):
        """Test ability to move a surface agent to touch an object.

        Given a substandard view of an object, the "experimenter" (via agent actions)
        can move a surface agent to touch the object before beginning the experiment.

        In this basic version, the object is a bit too far away, and so the agent
        moves forward
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.poor_initial_view_surf_agent_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.pre_episode()

            pprint("...stepping through observations...")

            # Get a first step to allow the surface agent to touch the object
            observation_pre_touch = next(exp.dataloader)
            exp.model.step(observation_pre_touch)

            # Check initial view post touch-attempt
            observation_post_touch = next(exp.dataloader)

            # TODO M remove the following train-wreck during refactor
            view = observation_post_touch[exp.model.motor_system._policy.agent_id][
                "view_finder"
            ]
            dict_config = config_to_dict(config)

            points_on_target_obj = (
                view["semantic_3d"][:, 3].reshape(view["depth"].shape) == 1
            )
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            assert closest_point_on_target_obj < 1.0, (
                f"Should be within a meter of the object, "
                f"closest point at {closest_point_on_target_obj}"
            )

            target_closest_point = dict_config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["desired_object_distance"]

            # Utility policy should not have moved too close to the object
            assert closest_point_on_target_obj > target_closest_point, (
                f"Initial position is too close, {closest_point_on_target_obj} "
                f"vs target of {target_closest_point}"
            )

    def test_get_good_view_multi_object(self):
        """Test ability to move a distant agent to a good view of an object.

        Given a substandard view of an object, the "experimenter" (via agent actions)
        can move a distant agent to a good view of the object before beginning the
        experiment.

        In this case, there are multiple objects, such that at the start of the
        experiment, the target object is not visible in the central pixel of the view.
        Thus the policy must both turn towards the target object (using the viewfinder),
        and move towards it.
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.poor_initial_view_multi_object_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()

            # Manually go through evaluation (i.e. methods in .evaluate()
            # and run_epoch())
            exp.model.set_experiment_mode("eval")
            exp.pre_epoch()
            exp.pre_episode()

            pprint("...stepping through observations...")
            # Check the initial view
            observation = next(exp.dataloader)
            # TODO M remove the following train-wreck during refactor
            view = observation[exp.model.motor_system._policy.agent_id]["view_finder"]
            semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
            perc_on_target_obj = get_perc_on_obj_semantic(semantic, semantic_id=1)

            dict_config = config_to_dict(config)
            target_perc_on_target_obj = dict_config["monty_config"][
                "motor_system_config"
            ]["motor_system_args"]["policy_args"]["good_view_percentage"]

            assert perc_on_target_obj >= target_perc_on_target_obj, (
                f"Initial view is not good enough, {perc_on_target_obj} "
                f"vs target of {target_perc_on_target_obj}"
            )

            points_on_target_obj = semantic == 1
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            target_closest_point = dict_config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["desired_object_distance"]

            # Utility policy should not have moved too close to the object
            assert closest_point_on_target_obj > target_closest_point, (
                f"Initial view is too close to target, {closest_point_on_target_obj}"
                f" vs target of {target_closest_point}"
            )

            # Also calculate closest point on *any* object so that we don't get
            # too close and clip into objects; NB that any object will have a
            # semantic ID > 0
            points_on_any_obj = view["semantic"] > 0
            closest_point_on_any_obj = np.min(view["depth"][points_on_any_obj])
            assert closest_point_on_any_obj > target_closest_point / 6, (
                f"Initial view too close to other objects, {closest_point_on_any_obj} "
                f"vs target of {target_closest_point / 6}"
            )

    def test_distant_policy_moves_back_to_object(self):
        """Test ability of distant agent to move back to an object.

        Test that the standard-distant agent policy (performing saccades) correctly
        moves back to the object after falling off.

        Uses an action policy with high-stickiness and large saccade sizes, so
        that we are guaranteed to move off of the cube.
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_action_distant_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            pprint("...training...")
            exp.pre_epoch()

            # Only do a single episode
            exp.pre_episode()

            pprint("...stepping through observations...")
            # Manually step through part of run_episode function
            for loader_step, observation in enumerate(exp.dataloader):
                exp.model.step(observation)

                last_action = exp.model.motor_system.last_action

                if loader_step == 3:
                    stored_action = last_action
                    assert not exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), "Should be off object"

                if loader_step == 4:
                    should_have_moved_back = (
                        "Should have moved back by reversing last movement"
                    )
                    self.assertIsInstance(
                        last_action, type(stored_action), should_have_moved_back
                    )
                    if isinstance(stored_action, (LookDown, LookUp)):
                        self.assertEqual(
                            last_action.rotation_degrees,
                            -stored_action.rotation_degrees,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.constraint_degrees,
                            stored_action.constraint_degrees,
                            should_have_moved_back,
                        )
                    elif isinstance(stored_action, (TurnLeft, TurnRight)):
                        self.assertEqual(
                            last_action.rotation_degrees,
                            -stored_action.rotation_degrees,
                            should_have_moved_back,
                        )
                    elif isinstance(stored_action, MoveForward):
                        self.assertEqual(
                            last_action.distance,
                            -stored_action.distance,
                            should_have_moved_back,
                        )
                    elif isinstance(stored_action, MoveTangentially):
                        self.assertEqual(
                            last_action.distance,
                            -stored_action.distance,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.direction,
                            stored_action.direction,
                            should_have_moved_back,
                        )
                    elif isinstance(stored_action, OrientHorizontal):
                        self.assertEqual(
                            last_action.rotation_degrees,
                            -stored_action.rotation_degrees,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.left_distance,
                            -stored_action.left_distance,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.forward_distance,
                            -stored_action.forward_distance,
                            should_have_moved_back,
                        )
                    elif isinstance(stored_action, OrientVertical):
                        self.assertEqual(
                            last_action.rotation_degrees,
                            -stored_action.rotation_degrees,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.down_distance,
                            -stored_action.down_distance,
                            should_have_moved_back,
                        )
                        self.assertEqual(
                            last_action.forward_distance,
                            -stored_action.forward_distance,
                            should_have_moved_back,
                        )
                    assert exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), "Should be back on object"
                    break  # Don't go into exploratory mode

    def test_surface_policy_moves_back_to_object(self):
        """Test ability of surface agent to move back to an object.

        Test that the standard surface-agent policy correctly moves back to the
        object after falling off.

        Uses an action policy with high-stickiness, so that we are guaranteed to move
        off of the cube.
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.fixed_action_surface_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            pprint("...training...")
            exp.pre_epoch()

            # Only do a single episode
            exp.pre_episode()

            pprint("...stepping through observations...")
            # Take several steps in a fixed direction until we fall off the object, then
            # ensure we get back on to it
            for loader_step, observation in enumerate(exp.dataloader):
                exp.model.step(observation)

                #  Step | Action           | Motor-only? | Obs processed? | Source
                # ------|------------------|-------------|----------------|-------------
                #  1    | MoveForward      | True        | False          | dynamic_call
                #  2    | OrientHorizontal | True        | False          | dynamic_call
                #  3    | OrientVertical   | False       | True           | dynamic_call
                #  4    | MoveTangentially | True        | False          | dynamic_call
                #  5    | MoveForward      | True        | False          | dynamic_call
                #  6    | OrientHorizontal | True        | False          | dynamic_call
                #  7    | OrientVertical   | False       | True           | dynamic_call
                #  8    | MoveTangentially | True        | False          | dynamic_call
                #  9    | MoveForward      | True        | False          | dynamic_call
                #  10   | OrientHorizontal | True        | False          | dynamic_call
                #  11   | OrientVertical   | False       | True           | dynamic_call
                #  12   | MoveTangentially | True        | False          | dynamic_call
                # falls off object
                #  13   | OrientHorizontal | True        | False          | touch_object
                #  14   | OrientHorizontal | True        | False          | touch_object
                #  15   | OrientHorizontal | True        | False          | touch_object
                #  16   | OrientHorizontal | True        | False          | touch_object
                #  17   | OrientHorizontal | True        | False          | touch_object
                #  18   | OrientHorizontal | True        | False          | touch_object
                #  19   | OrientHorizontal | True        | False          | touch_object
                #  20   | OrientHorizontal | True        | False          | touch_object
                #  21   | OrientHorizontal | True        | False          | touch_object
                #  22   | OrientHorizontal | True        | False          | touch_object
                #  23   | OrientHorizontal | True        | False          | touch_object
                #  24   | OrientHorizontal | True        | False          | touch_object
                #  25   | OrientVertical   | True        | False          | touch_object
                #  26   | MoveForward      | True        | False          | touch_object
                # back on object
                #  27   | MoveForward      | True        | False          | dynamic_call
                #  28   | OrientHorizontal | True        | False          | dynamic_call
                #  29   | OrientVertical   | False       | True           | dynamic_call
                #  30   | MoveTangentially | True        | False          | dynamic_call
                # falls off object
                #  31   | OrientHorizontal | True        | False          | touch_object
                #  32   | OrientHorizontal | True        | False          | touch_object
                #  33   | OrientHorizontal | True        | False          | touch_object
                #  34   | OrientHorizontal | True        | False          | touch_object
                #  35   | OrientHorizontal | True        | False          | touch_object
                #  36   | OrientHorizontal | True        | False          | touch_object
                #  37   | OrientHorizontal | True        | False          | touch_object
                #  38   | OrientHorizontal | True        | False          | touch_object
                #  39   | OrientHorizontal | True        | False          | touch_object
                #  40   | OrientHorizontal | True        | False          | touch_object
                #  41   | OrientHorizontal | True        | False          | touch_object
                #  42   | OrientHorizontal | True        | False          | touch_object
                #  43   | OrientVertical   | True        | False          | touch_object
                #  44   | MoveForward      | True        | False          | touch_object
                # back on object
                #  45   | MoveForward      | True        | False          | dynamic_call
                #  46   | OrientHorizontal | True        | False          | dynamic_call
                #  47   | OrientVertical   | False       | True           | dynamic_call
                #  48   | MoveTangentially | True        | False          | dynamic_call
                # falls off object
                #  49   | OrientHorizontal | True        | False          | touch_object
                #  50   | OrientHorizontal | True        | False          | touch_object
                #  51   | OrientHorizontal | True        | False          | touch_object
                #  52   | OrientHorizontal | True        | False          | touch_object
                #  53   | OrientHorizontal | True        | False          | touch_object
                #  54   | OrientHorizontal | True        | False          | touch_object
                #  56   | OrientHorizontal | True        | False          | touch_object
                #  57   | OrientHorizontal | True        | False          | touch_object
                #  58   | OrientHorizontal | True        | False          | touch_object
                #  59   | OrientHorizontal | True        | False          | touch_object
                #  60   | OrientHorizontal | True        | False          | touch_object
                #  61   | OrientVertical   | True        | False          | touch_object
                #  62   | MoveForward      | True        | False          | touch_object
                # back on object
                #  63   | MoveForward      | True        | False          | dynamic_call
                #  64   | OrientHorizontal | True        | False          | dynamic_call
                #  65   | OrientVertical   | False       | True           | dynamic_call
                #  66   | MoveTangentially | True        | False          | dynamic_call
                # falls off object
                #  67   | OrientHorizontal | True        | False          | touch_object

                # Motor-only touch_object steps
                if (
                    13 <= loader_step <= 26
                    or 31 <= loader_step <= 44
                    or 49 <= loader_step <= 62
                    or loader_step == 67
                ):
                    assert not exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), (
                        f"Should be off object, motor-only step: {loader_step}"
                    )
                if loader_step == 67:
                    break  # Finish test

                # First two on-object steps are always MoveForward & OrientHorizontal
                # motor-only steps
                if loader_step in [27, 28, 45, 46, 63, 64]:
                    assert not exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), (
                        f"Should be on object, motor-only step: {loader_step}"
                    )

                # Third on-object steps are always OrientVertical that send data to LM
                if loader_step in [29, 47, 65]:
                    assert exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), (
                        f"Should be on object, sending data to LM: {loader_step}"
                    )

                # Fourth on-object steps are always MoveTangentially motor-only steps
                if loader_step in [30, 48, 66]:
                    assert not exp.model.learning_modules[
                        0
                    ].buffer.get_last_obs_processed(), (
                        f"Should be on object, motor-only step: {loader_step}"
                    )

    def test_surface_policy_orientation(self):
        """Test ability of surface agent to orient to a surface normal.

        Test that the surface-agent correctly orients to be pointing down at an
        observed surface normal.

        Begins the episode by facing a cube whose surface is pointing away from
        the agent at an odd angle.
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.rotated_cube_view_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            pprint("...training...")
            exp.pre_epoch()
            exp.pre_episode()

            pprint("...stepping through observations...")
            for loader_step, observation in enumerate(exp.dataloader):
                exp.model.step(observation)
                exp.post_step(loader_step, observation)

                if loader_step == 3:  # Surface agent should have re-oriented
                    break

            # Most recently observed surface normal sent to the learning module
            current_pose = exp.model.learning_modules[0].buffer.get_current_pose(
                input_channel="first"
            )

            # Rotate vector representing agent's pointing direction by the agent's
            # current orientation
            agent_direction = np.array(
                hab_utils.quat_rotate_vector(
                    exp.model.motor_system._state["agent_id_0"]["rotation"],
                    [
                        0,
                        0,
                        -1,
                    ],  # The initial direction vector corresponding to the agent's
                    # orientation
                )
            )

            assert np.all(
                np.isclose(
                    current_pose[1], agent_direction * (-1), rtol=1.0e-3, atol=1.0e-2
                )
            ), "Agent should be (approximately) looking down on the surface normal"

    def test_core_following_principal_curvature(self):
        """Test ability of surface agent to follow principal curvature.

        Test that the surface-agent follows the principal curvature direction when
        the PC information is present.

        This basic unit test checks that we follow the minimal and then maximal
        principal curvature for a number of steps, including several other basic
        settings that can arise.

        Note these movements are not actually performed, i.e. they represent
        hypothetical outputs from the motor-system.
        """
        motor_system, motor_system_args = self.initialize_motor_system(
            MotorSystemConfigCurvatureInformedSurface(
                motor_system_args=dict(
                    policy_class=SurfacePolicyCurvatureInformed,
                    policy_args=make_curv_surface_policy_config(
                        desired_object_distance=0.025,
                        alpha=0.1,
                        pc_alpha=0.5,
                        max_pc_bias_steps=2,  # Overwrite default value
                        min_general_steps=8,
                        min_heading_steps=12,
                        use_goal_state_driven_actions=False,
                    ),
                ),
            )
        )

        # Initialize motor-system state
        motor_system._state = dict(agent_id_0={})
        motor_system._state["agent_id_0"]["rotation"] = qt.quaternion(1, 0, 0, 0)

        # Step 1
        # fake_obs_pc contains observations including the surface normal and principal
        # curvature directions in the global/environment reference frame; the movement
        # (specifically tangential translation) that the agent should take is
        # also in environmental coordinates, so we compare these
        # Note that the movement is a unit vector because it is a direction, the amount
        # (i.e. size) of the translation is represented separately.
        motor_system._policy.processed_observations = self.fake_obs_pc[0]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 1, (
            "Should have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 1, (
            "Should have incremented continuous counter"
        )

        # Step 2
        motor_system._policy.processed_observations = self.fake_obs_pc[1]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 3: Our bias should change from following minimal to maximal
        # PC
        motor_system._policy.processed_observations = self.fake_obs_pc[2]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [0, 1, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 1, (
            "Should have reset following PC counter due to bias change, and incremented"
        )
        assert motor_system._policy.continuous_pc_steps == 1, (
            "Should have reset continous counter due to bias change, and incremented"
        )

        # Step 4
        motor_system._policy.processed_observations = self.fake_obs_pc[3]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [0, 1, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 5: Pass observation *without* a well defined PC direction
        motor_system._policy.processed_observations = self.fake_obs_pc[4]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        # Note the following movement is a random direction deterministcally set by the
        # random seed
        assert np.all(np.isclose(direction, [-0.13745981, 0.99050735, 0])), (
            "Not following correct non-PC direction"
        )
        assert motor_system._policy.ignoring_pc_counter == 1, (
            "Should have reset ignoring_pc_counter, and then incremented"
        )
        assert motor_system._policy.continuous_pc_steps == 0, (
            "Should have reset continuous counter"
        )
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert motor_system._policy.using_pc_guide is False, (
            "Should not be using PC guide"
        )
        assert motor_system._policy.prev_angle is None, "Should have reset prev_angle"

        # Step 6 : Follow principal curvature, but the agent is rotated, so the policy
        # needs to ensure PC is still handled correctly (PC and the returned movement
        # vector are both in environment coordinates, so in effect the result should be
        # the same); note the agent is still orthogonal to the PC directions.

        # Update relevant motor-system variables
        motor_system._policy.ignoring_pc_counter = motor_system_args["policy_args"][
            "min_general_steps"
        ]
        motor_system._state["agent_id_0"]["rotation"] = qt.quaternion(0, 0, 1, 0)

        motor_system._policy.processed_observations = self.fake_obs_pc[5]
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [1.0, 0.0, 0])), (
            "Not following correct PC direction"
        )

    def test_advanced_following_principal_curvature(self):
        """Test more edge-case elements of the following-PC policy.

        This unit test checks more edge-case elements of the following-PC policy,
        such as checks to avoid doubling back on ourself, and how to handle when the
        proposed PC points in the z direction (i.e. towards or away from the agent).
        """
        motor_system, motor_system_args = self.initialize_motor_system(
            MotorSystemConfigCurvatureInformedSurface(
                motor_system_args=dict(
                    policy_class=SurfacePolicyCurvatureInformed,
                    policy_args=make_curv_surface_policy_config(
                        desired_object_distance=0.025,
                        alpha=0.1,
                        pc_alpha=0.5,
                        max_pc_bias_steps=32,
                        min_general_steps=1,  # Overwrite default value so that we more
                        # quickly transition into taking PC steps when testing this
                        min_heading_steps=12,
                        use_goal_state_driven_actions=False,
                    ),
                ),
            )
        )

        # Initialize motor system state
        motor_system._state = dict(agent_id_0={})
        motor_system._state["agent_id_0"]["rotation"] = qt.quaternion(1, 0, 0, 0)

        # Step 1 : PC-guided information, but we haven't taken the minimum number of
        # non-PC steps, so take random step
        motor_system._policy.ignoring_pc_counter = 0  # Set to 0 so we skip PC
        motor_system._policy.processed_observations = self.fake_obs_advanced_pc[0]
        # TODO M clean up how we set this when doing the refactor; currently this is
        # done in graph_matching.py normally
        motor_system._policy.tangent_locs.append(self.fake_obs_pc[0].location)
        motor_system._policy.tangent_norms.append([0, 0, 1])
        direction = motor_system._policy.tangential_direction(motor_system._state)
        # Note the following movement is a random direction deterministcally set by the
        # random seed
        assert np.all(np.isclose(direction, [0.98165657, 0.19065773, 0])), (
            "Not following correct non-PC direction"
        )
        assert motor_system._policy.following_pc_counter == 0, (
            "Should not have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 0, (
            "Should not have incremented continuous counter"
        )

        # Step 2 : Given the same observation, but now have taken sufficient non-PC
        # steps, so should follow PC direction
        motor_system._policy.processed_observations = self.fake_obs_advanced_pc[0]
        # TODO M clean up how we set this when doing the refactor; currently this is
        # done in graph_matching.py normally
        motor_system._policy.tangent_locs.append(self.fake_obs_pc[0].location)
        motor_system._policy.tangent_norms.append([0, 0, 1])
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 1, (
            "Should have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 1, (
            "Should have incremented continuous counter"
        )

        # Step 3 : Following PC direction would cause us to double back on ourself;
        # PC has been aribtrarily flipped vs. previous step, so can just flip it back
        motor_system._policy.processed_observations = self.fake_obs_advanced_pc[1]
        motor_system._policy.tangent_locs.append(self.fake_obs_advanced_pc[1].location)
        motor_system._policy.tangent_norms.append([0, 0, 1])
        direction = motor_system._policy.tangential_direction(motor_system._state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert motor_system._policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 4 : PC is defined in z-direction, so policy should take a random step
        motor_system._policy.processed_observations = self.fake_obs_advanced_pc[2]
        motor_system._policy.tangent_locs.append(self.fake_obs_pc[2].location)
        motor_system._policy.tangent_norms.append([0, 0, 1])
        direction = motor_system._policy.tangential_direction(motor_system._state)
        # Note the following movement is a random direction deterministcally set by the
        # random seed
        assert np.all(
            np.isclose(direction, [0.9789808522232504, -0.20395217816987962, 0])
        ), "Not following correct non-PC direction"
        assert (
            motor_system._policy.ignoring_pc_counter
            == motor_system_args["policy_args"]["min_general_steps"]
        ), "Shoulnd't increment ignoring_pc_counter"
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert motor_system._policy.pc_is_z_defined is True, (
            "Should have detected z-defined PC"
        )

        # Step 5 : Following PC direction would cause us to double back on ourself; PC
        # has not been arbitrarily flipped, so policy selects a new heading
        motor_system._policy.processed_observations = self.fake_obs_advanced_pc[0]
        motor_system._policy.tangent_locs.append(
            self.fake_obs_pc[0].location
        )  # Synthetically
        # "teleport" the agent back to the first observation and location, such that
        # following PC would cause it to visit the observation 1 again (which it is
        # designed to avoid)
        motor_system._policy.tangent_norms.append([0, 0, 1])
        direction = motor_system._policy.tangential_direction(motor_system._state)
        # Note the following movement is a random direction deterministcally set by the
        # random seed
        assert np.all(np.isclose(direction, [0.60958557, 0.79272027, 0])), (
            "Not following correct non-PC direction"
        )
        assert motor_system._policy.ignoring_pc_counter == 0, (
            "Should have reset ignoring_pc_counter, and not incremented"
        )
        assert motor_system._policy.continuous_pc_steps == 0, (
            "Should have reset continuous counter"
        )
        assert motor_system._policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert motor_system._policy.using_pc_guide is False, (
            "Should not be using PC guide"
        )
        assert motor_system._policy.prev_angle is None, "Should have reset prev_angle"
        assert motor_system._policy.pc_is_z_defined is False, (
            "Should have reset z-defind flag"
        )

    def core_evaluate_compute_goal_state_for_target_loc(
        self, lm, motor_system, object_orientation, target_location_on_object
    ):
        """Test GSGs ability to propose a motor-system goal-state.

        Test the GSGs ability to propose a motor-system goal-state, and then for
        the motor-system to propose a particular target agent location and
        orientation in Habitat-compatible coordinates.

        Args:
            lm: The LM with the GSG that we will test
            motor_system: The motor-system to test
            object_orientation: The orientation of the object in Euler angle degrees
            target_location_on_object: The location in object-centric coordinates
                which the agent should move to

        Returns:
            motor_goal_state_location: Motor goal-state location
            motor_goal_state_pose: Motor goal-state 0th pose vector
            target_loc_hab: Habitat target location
            agent_direction_hab: Habitat agent direction
        """
        # --- Determine the motor-goal state ---

        target_info = {
            "target_loc": np.array(target_location_on_object),
            "target_surface_normal": np.array([0.0, 1.0, 0.0]),
            "hypothesis_to_test": {
                "graph_id": "dummy_object",
                "location": np.array([0.1, 0.1, 0.1]),
                "rotation": Rotation.from_euler(
                    "xyz", object_orientation, degrees=True
                ).inv(),  # Rotation to transform the feature
                "scale": 1,
                "evidence": 100,
            },
        }

        fake_sensation_config = dict(
            location=np.array([0, 1.5, 0.1]),
            morphological_features={
                "pose_vectors": np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures": [-5, 10],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

        lm.pre_episode(
            primary_target=dict(
                object="dummy_object",
                quat_rotation=[1.0, 0.0, 0.0, 0.0],  # Filler value
            )
        )

        lm.matching_step(observations=[State(**fake_sensation_config)])

        # GSG handles computing the motor goal-state
        motor_goal_state = lm.gsg._compute_goal_state_for_target_loc(
            observations=[State(**fake_sensation_config)],
            target_info=target_info,
        )

        # --- Determine Habitat-coordinates from goal-state ---

        motor_system._policy.set_driving_goal_state(motor_goal_state)

        target_loc_hab, target_quat = motor_system._policy.derive_habitat_goal_state()

        resulting_rot = Rotation.from_quat(
            numpy_to_scipy_quat(np.array([target_quat.real] + list(target_quat.imag)))
        )

        # As the agent faces "forward" along the negative z-axis, we use this vector
        # to visualize its orientation
        agent_direction_hab = resulting_rot.apply(np.array([0, 0, -1]))

        return (
            motor_goal_state.location,
            motor_goal_state.morphological_features["pose_vectors"][0],
            target_loc_hab,
            agent_direction_hab,
        )

    def test_multi_param_compute_goal_state_for_target_loc(self):
        """Perform core_evaluate_compute_goal_state_for_target_loc.

        Should work across a variety of parameter settings.
        """
        lm, gsg_args = self.initialize_lm_with_gsg()

        motor_system, _ = self.initialize_motor_system(
            MotorSystemConfigCurvatureInformedSurface()
        )

        # The target displacement of the agent from the object; used to determine
        # the validity of the final agent location
        surface_displacement = gsg_args["desired_object_distance"] * 1.5

        # === First, easy example ===
        (
            motor_goal_location,
            motor_goal_direction,
            target_loc_hab,
            agent_direction_hab,
        ) = self.core_evaluate_compute_goal_state_for_target_loc(
            lm,
            motor_system,
            object_orientation=[0, 0, 0],
            target_location_on_object=[0.2, 0.2, 0.2],
        )

        assert np.all(
            np.isclose(motor_goal_location, [0.1, 1.6 + surface_displacement, 0.2])
        ), "Goal-state location is not as expected"

        # Pointing down
        assert np.all(np.isclose(motor_goal_direction, [0, -1.0, 0])), (
            "Goal-state pose is not as expected"
        )

        assert np.all(
            np.isclose(target_loc_hab, [0.1, 1.6 + surface_displacement, 0.2])
        ), "Habitat target location is not as expected"

        # Pointing down
        assert np.all(np.isclose(agent_direction_hab, [0, -1.0, 0])), (
            "Habitat pose is not as expected"
        )

        # === Second, harder example ===

        (
            motor_goal_location_2,
            motor_goal_direction_2,
            target_loc_hab_2,
            agent_direction_hab_2,
        ) = self.core_evaluate_compute_goal_state_for_target_loc(
            lm,
            motor_system,
            object_orientation=[180, 0, 0],  # Flip the object around the x-axis, such
            # that e.g. a vector pointing up will now point down
            target_location_on_object=[0.1, 0.2, 0.1],
        )

        # Surface displacement is negative, because object is flipped in x-axis
        assert np.all(
            np.isclose(motor_goal_location_2, [0, 1.4 - surface_displacement, 0.1])
        ), "Goal-state location is not as expected"

        # Pointing up, because object is flipped in y-axis
        assert np.all(np.isclose(motor_goal_direction_2, [0, 1.0, 0])), (
            "Goal-state pose is not as expected"
        )

        # Surface displacement is negative, because object is flipped in x-axis
        assert np.all(
            np.isclose(target_loc_hab_2, [0, 1.4 - surface_displacement, 0.1])
        ), "Habitat target location is not as expected"

        # Pointing up, because object is flipped in y-axis
        assert np.all(np.isclose(agent_direction_hab_2, [0, 1.0, 0])), (
            "Habitat pose is not as expected"
        )

        # === Third, hardest example ===

        (
            motor_goal_location_3,
            motor_goal_direction_3,
            target_loc_hab_3,
            agent_direction_hab_3,
        ) = self.core_evaluate_compute_goal_state_for_target_loc(
            lm,
            motor_system,
            object_orientation=[160, 45, 70],
            target_location_on_object=[0.3, 0.2, 0.15],
        )

        # Below results manually verified
        assert np.all(
            np.isclose(motor_goal_location_3, [0.18586463, 1.58288073, -0.04139085])
        ), "Goal-state location is not as expected"

        assert np.all(
            np.isclose(motor_goal_direction_3, [-0.965738, 0.09413407, -0.24184476])
        ), "Goal-state pose is not as expected"

        assert np.all(
            np.isclose(target_loc_hab_3, [0.18586463, 1.58288073, -0.04139085])
        ), "Habitat target location is not as expected"

        assert np.all(
            np.isclose(agent_direction_hab_3, [-0.965738, 0.09413407, -0.24184476])
        ), "Habitat pose is not as expected"


if __name__ == "__main__":
    unittest.main()
