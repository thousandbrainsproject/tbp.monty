# Copyright 2025-2026 Thousand Brains Project
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

import habitat_sim.utils as hab_utils
import hydra
import numpy as np
import quaternion as qt
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

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
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.motor_policies import (
    get_perc_on_obj_semantic,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.transform_utils import numpy_to_scipy_quat


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()

        def hydra_config(test_name: str) -> DictConfig:
            return hydra.compose(
                config_name="test",
                overrides=[
                    f"test=policy/{test_name}",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.base_dist_cfg = hydra_config("base_dist")
            self.base_surf_cfg = hydra_config("base_surf")
            self.spiral_cfg = hydra_config("spiral")
            self.curve_informed_cfg = hydra_config("curve_informed")
            self.surf_hypo_driven_cfg = hydra_config("surf_hypo_driven")
            self.dist_hypo_driven_cfg = hydra_config("dist_hypo_driven")
            self.dist_hypo_driven_multi_lm_cfg = hydra_config(
                "dist_hypo_driven_multi_lm"
            )
            self.dist_poor_initial_view_cfg = hydra_config("dist_poor_initial_view")
            self.surf_poor_initial_view_cfg = hydra_config("surf_poor_initial_view")
            self.poor_initial_view_multi_object_cfg = hydra_config(
                "poor_initial_view_multi_object"
            )
            self.dist_fixed_action_cfg = hydra_config("dist_fixed_action")
            self.surf_fixed_action_cfg = hydra_config("surf_fixed_action")
            self.rotated_cube_view_cfg = hydra_config("rotated_cube_view")

            self.motor_system_cfg_fragment = hydra.compose(
                config_name="experiment/config/monty/motor_system/curvature_informed_surface"
            ).experiment.config.monty.motor_system

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

        # No well-defined PC directions
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

        # PC direction "flipped", pointing back to a location we've already been at
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
        exp = hydra.utils.instantiate(self.base_dist_cfg.test)
        with exp:
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_spiral_policy(self):
        exp = hydra.utils.instantiate(self.spiral_cfg.test)
        with exp:
            # TODO: test that no two locations are the same
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_dist_agent_hypo_driven_policy(self):
        exp = hydra.utils.instantiate(self.dist_hypo_driven_cfg.test)
        with exp:
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_surface_policy(self):
        exp = hydra.utils.instantiate(self.base_surf_cfg.test)
        with exp:
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_curv_informed_policy(self) -> None:
        exp = hydra.utils.instantiate(self.curve_informed_cfg.test)
        with exp:
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_surf_agent_hypo_driven_policy(self):
        exp = hydra.utils.instantiate(self.surf_hypo_driven_cfg.test)
        with exp:
            exp.run()

    # @unittest.skip("debugging")
    def test_can_run_multi_lm_dist_agent_hypo_driven_policy(self):
        exp = hydra.utils.instantiate(self.dist_hypo_driven_multi_lm_cfg.test)
        with exp:
            exp.run()

    # ==== MORE INVOLVED TESTS OF ACTION POLICIES ====

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
            rng=np.random.RandomState(),
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
        exp = hydra.utils.instantiate(self.dist_poor_initial_view_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.pre_episode()

            # Check the initial view
            observation = next(exp.env_interface)
            # TODO M remove the following train-wreck during refactor
            view = observation[exp.model.motor_system._policy.agent_id]["view_finder"]
            semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
            perc_on_target_obj = get_perc_on_obj_semantic(semantic, semantic_id=1)

            config = self.dist_poor_initial_view_cfg.test.config

            target_perc_on_target_obj = config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["good_view_percentage"]

            assert perc_on_target_obj >= target_perc_on_target_obj, (
                f"Initial view is not good enough, {perc_on_target_obj} "
                f"vs target of {target_perc_on_target_obj}"
            )

            points_on_target_obj = semantic == 1
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            target_closest_point = config["monty_config"]["motor_system_config"][
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
        exp = hydra.utils.instantiate(self.surf_poor_initial_view_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.pre_episode()

            # Get a first step to allow the surface agent to touch the object
            observation_pre_touch = next(exp.env_interface)
            exp.model.step(observation_pre_touch)

            # Check initial view post touch-attempt
            observation_post_touch = next(exp.env_interface)

            # TODO M remove the following train-wreck during refactor
            view = observation_post_touch[exp.model.motor_system._policy.agent_id][
                "view_finder"
            ]

            config = self.surf_poor_initial_view_cfg.test.config

            points_on_target_obj = (
                view["semantic_3d"][:, 3].reshape(view["depth"].shape) == 1
            )
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            assert closest_point_on_target_obj < 1.0, (
                f"Should be within a meter of the object, "
                f"closest point at {closest_point_on_target_obj}"
            )

            target_closest_point = config["monty_config"]["motor_system_config"][
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
        Thus, the policy must both turn towards the target object (using the
        viewfinder), and move towards it.
        """
        exp = hydra.utils.instantiate(self.poor_initial_view_multi_object_cfg.test)
        with exp:
            exp.train()

            # Manually go through evaluation (i.e. methods in .evaluate()
            # and run_epoch())
            exp.experiment_mode = ExperimentMode.EVAL
            exp.model.set_experiment_mode("eval")
            exp.pre_epoch()
            exp.pre_episode()

            # Check the initial view
            observation = next(exp.env_interface)
            # TODO M remove the following train-wreck during refactor
            view = observation[exp.model.motor_system._policy.agent_id]["view_finder"]
            semantic = view["semantic_3d"][:, 3].reshape(view["depth"].shape)
            perc_on_target_obj = get_perc_on_obj_semantic(semantic, semantic_id=1)

            config = self.poor_initial_view_multi_object_cfg.test.config

            target_perc_on_target_obj = config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["good_view_percentage"]

            assert perc_on_target_obj >= target_perc_on_target_obj, (
                f"Initial view is not good enough, {perc_on_target_obj} "
                f"vs target of {target_perc_on_target_obj}"
            )

            points_on_target_obj = semantic == 1
            closest_point_on_target_obj = np.min(view["depth"][points_on_target_obj])

            target_closest_point = config["monty_config"]["motor_system_config"][
                "motor_system_args"
            ]["policy_args"]["desired_object_distance"]

            # Utility policy should not have moved too close to the object
            assert closest_point_on_target_obj > target_closest_point, (
                f"Initial view is too close to target, {closest_point_on_target_obj}"
                f" vs target of {target_closest_point}"
            )

            # Also calculate the closest point on *any* object so that we don't get
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
        exp = hydra.utils.instantiate(self.dist_fixed_action_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()

            # Only do a single episode
            exp.pre_episode()

            # Manually step through part of run_episode function
            for loader_step, observation in enumerate(exp.env_interface):
                exp.model.step(observation)

                last_action = exp.model.motor_system._policy.action

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
        exp = hydra.utils.instantiate(self.surf_fixed_action_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()

            # Only do a single episode
            exp.pre_episode()

            # Take several steps in a fixed direction until we fall off the object, then
            # ensure we get back on to it
            for loader_step, observation in enumerate(exp.env_interface):
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
        exp = hydra.utils.instantiate(self.rotated_cube_view_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.pre_episode()

            for loader_step, observation in enumerate(exp.env_interface):
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
                    exp.model.motor_system._state[AgentID("agent_id_0")].rotation,
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
        motor_system_cfg = OmegaConf.to_object(self.motor_system_cfg_fragment)
        policy_class = motor_system_cfg["motor_system_args"]["policy_class"]
        policy_args = motor_system_cfg["motor_system_args"]["policy_args"]
        policy_args["max_pc_bias_steps"] = 2
        rng = np.random.RandomState(123)
        policy = policy_class(rng=rng, **policy_args)
        policy.pre_episode(rng)

        # Initialize motor-system state
        proprioceptive_state = ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    position=np.array([0, 0, 0]),  # unused
                    rotation=qt.quaternion(1, 0, 0, 0),
                    sensors={},  # unused
                )
            }
        )

        # Step 1
        # fake_obs_pc contains observations including the surface normal and principal
        # curvature directions in the global/environment reference frame; the movement
        # (specifically tangential translation) that the agent should take is
        # also in environmental coordinates, so we compare these
        # Note that the movement is a unit vector because it is a direction, the amount
        # (i.e. size) of the translation is represented separately.
        policy.processed_observations = self.fake_obs_pc[0]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 1, (
            "Should have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 1, (
            "Should have incremented continuous counter"
        )

        # Step 2
        policy.processed_observations = self.fake_obs_pc[1]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 3: Our bias should change from following minimal to maximal
        # PC
        policy.processed_observations = self.fake_obs_pc[2]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [0, 1, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 1, (
            "Should have reset following PC counter due to bias change, and incremented"
        )
        assert policy.continuous_pc_steps == 1, (
            "Should have reset continous counter due to bias change, and incremented"
        )

        # Step 4
        policy.processed_observations = self.fake_obs_pc[3]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [0, 1, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 5: Pass observation *without* a well-defined PC direction
        policy.processed_observations = self.fake_obs_pc[4]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.isclose(
            np.dot(self.fake_obs_pc[4].get_surface_normal(), direction), 0
        ), "Direction should be orthogonal to tangent (surface) plane"
        assert policy.ignoring_pc_counter == 1, (
            "Should have reset ignoring_pc_counter, and then incremented"
        )
        assert policy.continuous_pc_steps == 0, "Should have reset continuous counter"
        assert policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert policy.using_pc_guide is False, "Should not be using PC guide"
        assert policy.prev_angle is None, "Should have reset prev_angle"

        # Step 6 : Follow principal curvature, but the agent is rotated, so the policy
        # needs to ensure PC is still handled correctly (PC and the returned movement
        # vector are both in environment coordinates, so in effect the result should be
        # the same); note the agent is still orthogonal to the PC directions.

        # Update relevant motor-system variables
        policy.ignoring_pc_counter = policy_args["min_general_steps"]
        proprioceptive_state[AgentID("agent_id_0")].rotation = qt.quaternion(0, 0, 1, 0)

        policy.processed_observations = self.fake_obs_pc[5]
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [1.0, 0.0, 0])), (
            "Not following correct PC direction"
        )

    def test_advanced_following_principal_curvature(self):
        """Test more edge-case elements of the following-PC policy.

        This unit test checks more edge-case elements of the following-PC policy,
        such as checks to avoid doubling back on ourself, and how to handle when the
        proposed PC points in the z direction (i.e. towards or away from the agent).
        """
        motor_system_cfg = OmegaConf.to_object(self.motor_system_cfg_fragment)

        policy_class = motor_system_cfg["motor_system_args"]["policy_class"]
        policy_args = motor_system_cfg["motor_system_args"]["policy_args"]
        # Overwrite min_general_steps default value so that we more quickly transition
        # into taking PC steps when testing this
        policy_args["min_general_steps"] = 1
        rng = np.random.RandomState(123)
        policy = policy_class(rng=rng, **policy_args)
        policy.pre_episode(rng)

        # Initialize motor system state
        proprioceptive_state = ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    position=np.array([0, 0, 0]),  # unused
                    rotation=qt.quaternion(1, 0, 0, 0),
                    sensors={},  # unused
                )
            }
        )

        # Step 1 : PC-guided information, but we haven't taken the minimum number of
        # non-PC steps, so take random step
        policy.ignoring_pc_counter = 0  # Set to 0 so we skip PC
        policy.processed_observations = self.fake_obs_advanced_pc[0]
        # TODO M clean up how we set this when doing the refactor; currently this is
        # done in graph_matching.py normally
        policy.tangent_locs.append(self.fake_obs_advanced_pc[0].location)
        policy.tangent_norms.append([0, 0, 1])
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.isclose(
            np.dot(self.fake_obs_advanced_pc[0].get_surface_normal(), direction), 0
        ), "Direction should be orthogonal to tangent (surface) plane"
        assert policy.following_pc_counter == 0, (
            "Should not have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 0, (
            "Should not have incremented continuous counter"
        )

        # Step 2 : Given the same observation, but now have taken sufficient non-PC
        # steps, so should follow PC direction
        policy.processed_observations = self.fake_obs_advanced_pc[0]
        # TODO M clean up how we set this when doing the refactor; currently this is
        # done in graph_matching.py normally
        policy.tangent_locs.append(self.fake_obs_pc[0].location)
        policy.tangent_norms.append([0, 0, 1])
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 1, (
            "Should have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 1, (
            "Should have incremented continuous counter"
        )

        # Step 3 : Following PC direction would cause us to double back on ourself;
        # PC has been arbitrarily flipped vs. previous step, so can just flip it back
        policy.processed_observations = self.fake_obs_advanced_pc[1]
        policy.tangent_locs.append(self.fake_obs_advanced_pc[1].location)
        policy.tangent_norms.append([0, 0, 1])
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.all(np.isclose(direction, [1, 0, 0])), (
            "Not following correct PC direction"
        )
        assert policy.following_pc_counter == 2, (
            "Should have followed PC and incremented counter"
        )
        assert policy.continuous_pc_steps == 2, (
            "Should have incremented continuous counter"
        )

        # Step 4 : PC is defined in z-direction, so policy should take a random step
        policy.processed_observations = self.fake_obs_advanced_pc[2]
        policy.tangent_locs.append(self.fake_obs_advanced_pc[2].location)
        policy.tangent_norms.append([0, 0, 1])
        direction = policy.tangential_direction(proprioceptive_state)
        assert np.isclose(np.linalg.norm(direction), 1), (
            "Direction should be a unit vector"
        )
        assert np.isclose(direction[2], 0), (
            "Direction should be in the x-y plane (relative to the agent)"
        )
        assert policy.ignoring_pc_counter == policy_args["min_general_steps"], (
            "Shouldn't increment ignoring_pc_counter"
        )
        assert policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert policy.pc_is_z_defined is True, "Should have detected z-defined PC"

        # Step 5 : Following PC direction would cause us to double back on ourself; PC
        # has not been arbitrarily flipped, so policy selects a new heading
        policy.processed_observations = self.fake_obs_advanced_pc[0]
        policy.tangent_locs.append(
            self.fake_obs_advanced_pc[0].location
        )  # Synthetically
        # "teleport" the agent back to the first observation and location, such that
        # following PC would cause it to visit the observation 1 again (which it is
        # designed to avoid)
        policy.tangent_norms.append([0, 0, 1])
        direction = policy.tangential_direction(proprioceptive_state)
        # Note the following movement is a random direction deterministically set by the
        # random seed
        assert np.isclose(
            np.dot(self.fake_obs_advanced_pc[0].get_surface_normal(), direction), 0
        ), "Direction should be orthogonal to tangent (surface) plane"
        assert policy.ignoring_pc_counter == 0, (
            "Should have reset ignoring_pc_counter, and not incremented"
        )
        assert policy.continuous_pc_steps == 0, "Should have reset continuous counter"
        assert policy.following_pc_counter == 2, (
            "Should have not changed following_pc_counter"
        )
        assert policy.using_pc_guide is False, "Should not be using PC guide"
        assert policy.prev_angle is None, "Should have reset prev_angle"
        assert policy.pc_is_z_defined is False, "Should have reset z-defind flag"

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
            rng=np.random.RandomState(),
            primary_target=dict(
                object="dummy_object",
                quat_rotation=[1.0, 0.0, 0.0, 0.0],  # Filler value
            ),
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

        motor_system_cfg = OmegaConf.to_object(self.motor_system_cfg_fragment)
        motor_system_class = motor_system_cfg["motor_system_class"]
        motor_system_args = motor_system_cfg["motor_system_args"]
        policy_class = motor_system_args["policy_class"]
        policy_args = motor_system_args["policy_args"]
        rng = np.random.RandomState(123)
        policy = policy_class(rng=rng, **policy_args)
        motor_system = motor_system_class(policy=policy)
        motor_system.pre_episode(rng)

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
