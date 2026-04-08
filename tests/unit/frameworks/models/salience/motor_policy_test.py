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

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal
from tbp.monty.frameworks.sensors import SensorID

AGENT = AgentID("agent_id_0")
SENSOR = SensorID("sensor_id_0")


class LookAtGoalTest(unittest.TestCase):
    def test_raises_error_if_no_goal_is_provided(self):
        policy = LookAtGoal(AGENT, SENSOR)
        with self.assertRaises(RuntimeError):
            policy(
                ctx=None,
                observations=None,
                state=MotorSystemState(),
                percept=None,
                goal=None,
            )
