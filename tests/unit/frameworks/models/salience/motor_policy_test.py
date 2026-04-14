# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import Mock

import numpy as np
import quaternion as qt
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.cmp import Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import LookUp, TurnLeft
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.salience.motor_policy import (
    GoalCollocatedWithSensor,
    LookAtGoal,
    NoGoalProvided,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tbp.monty.math import VectorXYZ

AGENT_ID = AgentID("agent_id_0")
FORWARD_VECTOR_REL_WORLD: VectorXYZ = (0.0, 0.0, -1.0)
SENSOR_ID = SensorID("sensor_id_0")
TEST_VOLUME_DISTANCE_METERS = 100.0


class LookAtGoalTest(unittest.TestCase):
    """This test class makes some simplifying assumptions.

    - The agent starts at the origin with identity rotation.
    - The sensor also starts at the origin with identity rotation.
    - Note that the agent and the sensor are collocated.
    """

    def setUp(self):
        identity_pose = {
            "position": (0, 0, 0),
            "rotation": qt.quaternion(1, 0, 0, 0),
        }
        self.sensor_state = SensorState(**identity_pose)
        self.agent_state = AgentState(
            sensors={SENSOR_ID: self.sensor_state}, **identity_pose
        )
        self.motor_system_state = MotorSystemState({AGENT_ID: self.agent_state})

    def test_raises_error_if_no_goal_is_provided_if_not_suppressing_runtime_errors(
        self,
    ) -> None:
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        with self.assertRaises(NoGoalProvided):
            policy(
                ctx=RuntimeContext(
                    rng=np.random.RandomState(42), suppress_runtime_errors=False
                ),
                observations=Mock(),
                state=MotorSystemState(),
                percept=Mock(),
                goal=None,
            )

    def test_returns_empty_result_if_no_goal_is_provided_if_suppressing_runtime_errors(
        self,
    ) -> None:
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        result = policy(
            ctx=RuntimeContext(
                rng=np.random.RandomState(42), suppress_runtime_errors=True
            ),
            observations=Mock(),
            state=MotorSystemState(),
            percept=Mock(),
            goal=None,
        )
        self.assertEqual(result.actions, [])

    def test_raises_error_if_goal_collocated_with_sensor_and_not_suppressing_runtime_errors(  # noqa: E501
        self,
    ) -> None:
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        with self.assertRaises(GoalCollocatedWithSensor):
            policy(
                ctx=RuntimeContext(
                    rng=np.random.RandomState(42), suppress_runtime_errors=False
                ),
                observations=Mock(),
                state=self.motor_system_state,
                percept=Mock(),
                goal=Goal(
                    location=np.array([0, 0, 0]),
                    morphological_features=None,
                    non_morphological_features=None,
                    confidence=1.0,
                    use_state=True,
                    sender_id="test",
                    sender_type="SM",
                    goal_tolerances=None,
                    info=None,
                ),
            )

    def test_returns_empty_result_if_goal_collocated_with_sensor_and_suppressing_runtime_errors(  # noqa: E501
        self,
    ) -> None:
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        result = policy(
            ctx=RuntimeContext(
                rng=np.random.RandomState(42), suppress_runtime_errors=True
            ),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=Goal(
                location=np.array([0, 0, 0]),
                morphological_features=None,
                non_morphological_features=None,
                confidence=1.0,
                use_state=True,
                sender_id="test",
                sender_type="SM",
                goal_tolerances=None,
                info=None,
            ),
        )
        self.assertEqual(result.actions, [])

    @given(
        goal_xyz=st.tuples(
            st.floats(
                min_value=-TEST_VOLUME_DISTANCE_METERS,
                max_value=TEST_VOLUME_DISTANCE_METERS,
            ),
            st.floats(
                min_value=-TEST_VOLUME_DISTANCE_METERS,
                max_value=TEST_VOLUME_DISTANCE_METERS,
            ),
            st.floats(
                min_value=-TEST_VOLUME_DISTANCE_METERS,
                max_value=TEST_VOLUME_DISTANCE_METERS,
            ),
        )
    )
    def test_returns_turn_left_and_look_up_oriented_at_the_goal(self, goal_xyz) -> None:
        """This test comes with some caveats.

        It ignores cases for generated goals collocated with the sensor. Those cases are
        tested elsewhere.
        """
        goal = Goal(
            location=np.array(goal_xyz),
            morphological_features=None,
            non_morphological_features=None,
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
            goal_tolerances=None,
            info=None,
        )
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        sensor_pos_rel_world = self.sensor_state.position
        expected_forward_vector_rel_world = np.array(
            [
                goal_xyz[0] - sensor_pos_rel_world[0],
                goal_xyz[1] - sensor_pos_rel_world[1],
                goal_xyz[2] - sensor_pos_rel_world[2],
            ]
        )
        if np.isclose(np.linalg.norm(expected_forward_vector_rel_world), 0.0):
            return

        result = policy(
            ctx=RuntimeContext(
                rng=np.random.RandomState(42), suppress_runtime_errors=False
            ),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=goal,
        )

        self.assertEqual(len(result.actions), 2)
        first_action = result.actions[0]
        self.assertEqual(first_action.name, TurnLeft.action_name())
        turn_left = cast("TurnLeft", first_action)
        second_action = result.actions[1]
        self.assertEqual(second_action.name, LookUp.action_name())
        look_up = cast("LookUp", second_action)
        self.assertLessEqual(abs(look_up.rotation_degrees), look_up.constraint_degrees)

        # Monty uses the "right-up-backward" convention, so the forward direction
        # vector is [0, 0, -1].
        # See: https://thousandbrainsproject.readme.io/docs/conventions
        rotation = Rotation.from_euler(
            "xyz",
            [look_up.rotation_degrees, turn_left.rotation_degrees, 0],
            degrees=True,
        )
        actuated_vector_rel_world = rotation.apply(FORWARD_VECTOR_REL_WORLD)

        expected_forward_vector_rel_world = normalize(expected_forward_vector_rel_world)
        np.testing.assert_array_almost_equal(
            expected_forward_vector_rel_world,
            actuated_vector_rel_world,
            decimal=12,
        )
