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

import numpy as np
import quaternion as qt
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.actions.actions import LookUp, TurnLeft
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("agent_id_0")
SENSOR_ID = SensorID("sensor_id_0")


class LookAtGoalTest(unittest.TestCase):
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

    def test_raises_error_if_no_goal_is_provided(self):
        policy = LookAtGoal(AGENT_ID, SENSOR_ID)
        with self.assertRaises(RuntimeError):
            policy(
                ctx=None,
                observations=None,
                state=MotorSystemState(),
                percept=None,
                goal=None,
            )

    @given(
        goal_xyz=st.tuples(
            st.floats(min_value=-1.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
        )
    )
    def test_returns_turn_left_and_look_up_oriented_at_the_goal(self, goal_xyz):
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
        result = policy(
            ctx=None,
            observations=None,
            state=self.motor_system_state,
            percept=None,
            goal=goal,
        )
        self.assertEqual(len(result.actions), 2)
        first_action = result.actions[0]
        self.assertEqual(first_action.name, TurnLeft.action_name())
        turn_left = cast("TurnLeft", first_action)
        second_action = result.actions[1]
        self.assertEqual(second_action.name, LookUp.action_name())
        look_up = cast("LookUp", second_action)

        # TODO: The "forward" direction should be the same as the vector from the
        #       agent's position to the goal location.
