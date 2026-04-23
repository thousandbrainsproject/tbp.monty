# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import numpy.typing as npt
import quaternion as qt
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation
from unittest_parametrize import ParametrizedTestCase, parametrize

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    InformedPolicyRandomWalk,
    JumpToGoal,
    MotorPolicyResult,
    PolicyStatus,
    PredefinedPolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tbp.monty.math import DEFAULT_TOLERANCE, VectorXYZ
from tests.unit.frameworks.models.fakes.cmp import FakeMessage
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    nonzero_magnitude_vectors,
    quaternions,
    vectors_3d,
)

AGENT_ID = AgentID("agent_id_0")


class SurfacePolicyCurvatureInformedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AGENT_ID
        self.policy = SurfacePolicyCurvatureInformed(
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            desired_object_distance=0.025,
        )
        self.location = np.array([1.0, 2.0, 3.0])
        self.tangent_norm = np.array([0, 1, 0])
        self.percept = Message(
            location=self.location,
            morphological_features={
                "pose_vectors": np.array(
                    [self.tangent_norm.tolist(), [1, 0, 0], [0, 0, -1]]
                ),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures_log": [0, 0.5],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def test_appends_to_tangent_locs_and_tangent_norms_if_last_action_is_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        np.testing.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(len(self.policy.tangent_norms), 1)
        np.testing.assert_array_equal(self.policy.tangent_norms[0], self.tangent_norm)

    def test_appends_none_to_tangent_norms_if_last_action_is_orient_vertical_but_no_pose_vectors_in_state(  # noqa: E501
        self,
    ):
        del self.percept.morphological_features["pose_vectors"]
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        np.testing.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(self.policy.tangent_norms, [None])

    def test_does_not_append_to_tangent_locs_and_tangent_norms_if_last_action_is_not_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = LookUp(
            agent_id=self.agent_id, rotation_degrees=0
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(self.policy.tangent_locs, [])
        self.assertEqual(self.policy.tangent_norms, [])


class PredefinedPolicyReadActionFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AGENT_ID
        self.actions_file = Path(__file__).parent / "motor_policies_test_actions.jsonl"

    def test_read_action_file(self) -> None:
        # For this test, we write our own actions to a temporary file instead of
        # loading a file on disk. It's a better guarantee that we're loading the
        # actions exactly as expected.
        expected = [
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
            LookDown(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=10.0),
            LookUp(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
        ]
        with tempfile.TemporaryDirectory() as data_path:
            actions_file = Path(data_path) / "actions.jsonl"
            actions_file.write_text(
                "\n".join(json.dumps(a, cls=ActionJSONEncoder) for a in expected) + "\n"
            )
            loaded = PredefinedPolicy.read_action_file(actions_file)
            self.assertEqual(len(loaded), len(expected))
            for loaded_action, expected_action in zip(loaded, expected):
                self.assertEqual(dict(loaded_action), dict(expected_action))

    def test_cycles_continuously(self) -> None:
        policy = PredefinedPolicy(
            agent_id=self.agent_id,
            file_name=self.actions_file,
        )
        cycle_length = len(policy.action_list)
        ctx = RuntimeContext(rng=np.random.RandomState(42))
        observations = Observations()
        returned_actions: list[Action] = []
        for _ in range(2 * cycle_length):
            result = policy(ctx, observations, MotorSystemState(), FakeMessage(), None)
            assert len(result.actions) == 1, "Expected one action"
            returned_actions.append(result.actions[0])

        for i in range(cycle_length):
            first_occurrence = returned_actions[i]
            second_occurrence = returned_actions[i + cycle_length]
            self.assertEqual(first_occurrence, second_occurrence)


class JumpToGoalTest(ParametrizedTestCase):
    def setUp(self) -> None:
        self.agent_id = AGENT_ID
        self.policy = JumpToGoal(self.agent_id, SensorID("view_finder"))
        self.motor_system_state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors={
                        SensorID("sensor_id_0"): SensorState(
                            position=cast("VectorXYZ", (0, 0, 0)), rotation=qt.one
                        )
                    },
                    position=cast("VectorXYZ", (0, 0, 0)),
                    rotation=qt.one,
                )
            }
        )

    @given(
        goal_location=vectors_3d(min_value=-1, max_value=1, dtype=np.float64),
        goal_direction=nonzero_magnitude_vectors(
            min_value=-1, max_value=1, dtype=np.float64
        ),
        policy=st.builds(
            JumpToGoal,
            agent_id=st.just(AGENT_ID),
            sensor_id=st.just(SensorID("view_finder")),
        ),
    )
    def test_generates_actions_that_align_forward_axis_with_goal_direction(
        self,
        goal_location: np.ndarray,
        goal_direction: np.ndarray,
        policy: JumpToGoal,
    ) -> None:
        pose_vectors = np.zeros((3, 3))
        goal_direction = normalize(goal_direction)
        pose_vectors[0] = goal_direction

        goal = Goal(
            location=goal_location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
            goal_tolerances=None,
            info=None,
        )

        policy_result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=goal,
        )
        assert isinstance(policy_result, MotorPolicyResult)

        self.assertEqual(len(policy_result.actions), 2)
        set_agent_pose = policy_result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        set_sensor_rotation = policy_result.actions[1]
        assert isinstance(set_sensor_rotation, SetSensorRotation)

        np.testing.assert_array_equal(set_agent_pose.location, goal_location)
        rotation = Rotation.from_quat(
            [
                # TODO(tslominski-tbp): Needs update when we use QuaternionWXYZ like
                # we're supposed to.
                set_agent_pose.rotation_quat.x,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.y,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.z,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.w,  # type: ignore[attr-defined]
            ]
        )
        new_forward_axis = -rotation.as_matrix()[:, 2]
        np.testing.assert_allclose(
            new_forward_axis, goal_direction, atol=DEFAULT_TOLERANCE
        )

        # Sensor rotation must be identity.
        np.testing.assert_allclose(
            qt.as_float_array(set_sensor_rotation.rotation_quat),
            qt.as_float_array(qt.one),
            atol=DEFAULT_TOLERANCE,
        )

    @parametrize("has_post_jump_goal", [(True,), (False,)])
    @given(
        agent_position=vectors_3d(min_value=-1, max_value=1, dtype=np.float64),
        agent_rotation=quaternions(),
        sensor_rotation=quaternions(),
    )
    def test_returns_undo_actions_status_ready_if_undo_is_needed_regardless_of_new_goal(
        self,
        has_post_jump_goal: bool,
        agent_position: npt.NDArray[np.float64],
        agent_rotation: qt.quaternion,
        sensor_rotation: qt.quaternion,
    ) -> None:
        pre_jump_state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    position=cast("VectorXYZ", tuple(agent_position)),
                    rotation=agent_rotation,
                    sensors={
                        SensorID("sensor_id_0"): SensorState(
                            position=cast("VectorXYZ", (0, 0, 0)),
                            rotation=sensor_rotation,
                        ),
                    },
                ),
            },
        )
        goal = Mock(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
            },
        )
        with patch(
            "tbp.monty.frameworks.models.motor_policies.PositioningProcedure.depth_at_center"
        ) as depth_at_center_mock:
            depth_at_center_mock.return_value = 1.0
            policy = JumpToGoal(self.agent_id, SensorID("view_finder"))
            policy(
                ctx=Mock(),
                observations=Mock(),
                state=pre_jump_state,
                percept=Mock(),
                goal=goal,
            )

            post_jump_goal = (
                Mock(
                    location=np.zeros(3),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                    },
                )
                if has_post_jump_goal
                else None
            )

            observations = Mock()
            policy_result = policy(
                ctx=Mock(),
                observations=observations,
                state=self.motor_system_state,
                percept=Mock(),
                goal=post_jump_goal,
            )

        assert isinstance(policy_result, MotorPolicyResult)
        self.assertEqual(policy_result.status, PolicyStatus.READY)
        self.assertEqual(len(policy_result.actions), 2)
        set_agent_pose = policy_result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        set_sensor_rotation = policy_result.actions[1]
        assert isinstance(set_sensor_rotation, SetSensorRotation)

        agent_state = pre_jump_state[self.agent_id]
        np.testing.assert_array_equal(set_agent_pose.location, agent_state.position)
        np.testing.assert_array_equal(
            qt.as_float_array(set_agent_pose.rotation_quat),
            qt.as_float_array(agent_state.rotation),
        )
        sensor_state = agent_state.sensors[SensorID("sensor_id_0")]
        np.testing.assert_array_equal(
            qt.as_float_array(set_sensor_rotation.rotation_quat),
            qt.as_float_array(sensor_state.rotation),
        )

        depth_at_center_mock.assert_called_once_with(
            agent_id=self.agent_id,
            observations=observations,
            sensor_id=SensorID("view_finder"),
        )

    @given(
        goal_location=vectors_3d(min_value=-1, max_value=1, dtype=np.float64),
        goal_direction=nonzero_magnitude_vectors(
            min_value=-1, max_value=1, dtype=np.float64
        ),
        policy=st.builds(
            JumpToGoal,
            agent_id=st.just(AGENT_ID),
            sensor_id=st.just(SensorID("view_finder")),
        ),
    )
    @patch(
        "tbp.monty.frameworks.models.motor_policies.PositioningProcedure.depth_at_center",
        return_value=0.99,
    )
    def test_returns_new_jump_actions_status_in_progress_if_undo_is_not_needed_after_jump_and_goal_is_provided(  # noqa: E501
        self,
        depth_at_center_mock: Mock,
        goal_location: np.ndarray,
        goal_direction: np.ndarray,
        policy: JumpToGoal,
    ) -> None:
        pose_vectors = np.zeros((3, 3))
        goal_direction = normalize(goal_direction)
        pose_vectors[0] = goal_direction

        first_goal = Mock(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
            },
        )
        second_goal = Mock(
            location=goal_location,
            morphological_features={
                "pose_vectors": pose_vectors,
            },
        )

        policy(
            ctx=Mock(),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=first_goal,
        )
        observations = Mock()
        policy_result = policy(
            ctx=Mock(),
            observations=observations,
            state=self.motor_system_state,
            percept=Mock(),
            goal=second_goal,
        )
        assert isinstance(policy_result, MotorPolicyResult)
        self.assertEqual(policy_result.status, PolicyStatus.IN_PROGRESS)

        self.assertEqual(len(policy_result.actions), 2)
        set_agent_pose = policy_result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        set_sensor_rotation = policy_result.actions[1]
        assert isinstance(set_sensor_rotation, SetSensorRotation)

        np.testing.assert_array_equal(set_agent_pose.location, goal_location)
        rotation = Rotation.from_quat(
            [
                # TODO(tslominski-tbp): Needs update when we use QuaternionWXYZ like
                # we're supposed to.
                set_agent_pose.rotation_quat.x,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.y,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.z,  # type: ignore[attr-defined]
                set_agent_pose.rotation_quat.w,  # type: ignore[attr-defined]
            ]
        )
        new_forward_axis = -rotation.as_matrix()[:, 2]
        np.testing.assert_allclose(
            new_forward_axis, goal_direction, atol=DEFAULT_TOLERANCE
        )

        # Sensor rotation must be identity.
        np.testing.assert_allclose(
            qt.as_float_array(set_sensor_rotation.rotation_quat),
            qt.as_float_array(qt.one),
            atol=DEFAULT_TOLERANCE,
        )

        depth_at_center_mock.assert_called_once_with(
            agent_id=self.agent_id,
            observations=observations,
            sensor_id=SensorID("view_finder"),
        )

    @patch(
        "tbp.monty.frameworks.models.motor_policies.PositioningProcedure.depth_at_center",
        return_value=0.99,
    )
    def test_returns_no_actions_status_ready_if_undo_is_not_needed_after_jump_and_goal_is_none(  # noqa: E501
        self,
        depth_at_center_mock: Mock,
    ) -> None:
        goal = Mock(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
            },
        )

        policy = JumpToGoal(self.agent_id, SensorID("view_finder"))
        policy(
            ctx=Mock(),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=goal,
        )
        observations = Mock()

        policy_result = policy(
            ctx=Mock(suppress_runtime_errors=False),
            observations=observations,
            state=self.motor_system_state,
            percept=Mock(),
            goal=None,
        )
        assert isinstance(policy_result, MotorPolicyResult)
        self.assertEqual(policy_result.status, PolicyStatus.READY)
        self.assertEqual(len(policy_result.actions), 0)

        depth_at_center_mock.assert_called_once_with(
            agent_id=self.agent_id,
            observations=observations,
            sensor_id=SensorID("view_finder"),
        )


class InformedPolicyRandomWalkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AGENT_ID

    def test_returns_no_actions_status_ready_if_never_on_object(self) -> None:
        policy = InformedPolicyRandomWalk(self.agent_id, Mock())
        percept = Mock()
        percept.get_on_object.return_value = False
        result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )
        assert isinstance(result, MotorPolicyResult)
        self.assertEqual(result.actions, [])
        self.assertIs(result.status, PolicyStatus.READY)

        percept.get_on_object.assert_called_with()

        for _ in range(10):
            result = policy(
                ctx=Mock(),
                observations=Mock(),
                state=MotorSystemState(),
                percept=percept,
                goal=Mock(),
            )
            assert isinstance(result, MotorPolicyResult)
            self.assertEqual(result.actions, [])
            self.assertIs(result.status, PolicyStatus.READY)
            percept.get_on_object.assert_called_with()

    def test_returns_sampled_action_status_ready_when_on_object(self) -> None:
        action_sampler_mock = Mock()
        action = LookUp(agent_id=self.agent_id, rotation_degrees=90)
        action_sampler_mock.sample.return_value = action
        policy = InformedPolicyRandomWalk(self.agent_id, action_sampler_mock)
        percept = Mock()
        percept.get_on_object.return_value = True
        rng_mock = Mock()

        result = policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        percept.get_on_object.assert_called_once_with()
        action_sampler_mock.sample.assert_called_once_with(self.agent_id, rng_mock)
        assert isinstance(result, MotorPolicyResult)
        self.assertEqual(result.actions, [action])
        self.assertIs(result.status, PolicyStatus.READY)

    @patch("tbp.monty.frameworks.models.motor_policies.fixme_undo_last_action")
    def test_returns_undo_status_ready_when_off_object_after_on_object(
        self,
        fixme_undo_last_action_mock: Mock,
    ) -> None:
        action_sampler_mock = Mock()
        action_mock = Mock()
        undo_action_mock = Mock()
        fixme_undo_last_action_mock.return_value = undo_action_mock
        action_sampler_mock.sample.return_value = action_mock
        policy = InformedPolicyRandomWalk(self.agent_id, action_sampler_mock)
        percept = Mock()
        percept.get_on_object.return_value = True
        rng_mock = Mock()

        policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        fixme_undo_last_action_mock.assert_called_once_with(action_mock)

        percept.get_on_object.return_value = False
        result = policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        percept.get_on_object.assert_called_with()

        assert isinstance(result, MotorPolicyResult)
        self.assertEqual(result.actions, [undo_action_mock])
        self.assertIs(result.status, PolicyStatus.READY)

    @patch("tbp.monty.frameworks.models.motor_policies.fixme_undo_last_action")
    def test_returns_undo_of_undo_status_ready_when_off_object_after_off_object_after_on_object(  # noqa: E501
        self,
        fixme_undo_last_action_mock: Mock,
    ) -> None:
        action_sampler_mock = Mock()
        action_mock = Mock()
        undo_action_mock = Mock()
        undo_of_undo_action_mock = Mock()
        fixme_undo_last_action_mock.return_value = undo_action_mock
        action_sampler_mock.sample.return_value = action_mock
        policy = InformedPolicyRandomWalk(self.agent_id, action_sampler_mock)
        percept = Mock()
        percept.get_on_object.return_value = True
        rng_mock = Mock()

        policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        fixme_undo_last_action_mock.assert_called_once_with(action_mock)

        fixme_undo_last_action_mock.return_value = undo_of_undo_action_mock
        percept.get_on_object.return_value = False
        policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        fixme_undo_last_action_mock.assert_called_with(undo_action_mock)
        percept.get_on_object.assert_called_with()

        result = policy(
            ctx=Mock(rng=rng_mock),
            observations=Mock(),
            state=MotorSystemState(),
            percept=percept,
            goal=Mock(),
        )

        fixme_undo_last_action_mock.assert_called_with(undo_of_undo_action_mock)
        assert isinstance(result, MotorPolicyResult)
        self.assertEqual(result.actions, [undo_of_undo_action_mock])
        self.assertIs(result.status, PolicyStatus.READY)
