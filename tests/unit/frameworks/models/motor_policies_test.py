# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from typing import Dict

import numpy as np

from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import LookUp
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)


class BasePolicyTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.agent_id = f"agent_id_{self.rng.randint(0, 999_999_999)}"
        self.default_sensor_state: SensorState = {
            "position": (0.0, 0.0, 0.0),
            "rotation": (1.0, 0.0, 0.0, 0.0),
        }
        self.agent_sensors: Dict[str, SensorState] = {
            f"sensor_id_{self.rng.randint(0, 999_999_999)}": self.default_sensor_state,
        }
        self.default_agent_state: AgentState = {
            "sensors": self.agent_sensors,
            **self.default_sensor_state,
        }

        self.policy = BasePolicy(
            rng=self.rng,
            action_sampler_args=dict(actions=[LookUp]),
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=self.agent_id,
            switch_frequency=0.05,
        )

    def test_get_agent_state_selects_state_matching_agent_id(self):
        expected_state: AgentState = {
            "sensors": self.agent_sensors,
            **self.default_sensor_state,
        }
        state = MotorSystemState(
            {
                f"{self.agent_id}": expected_state,
                "different_agent_id": {},
            }
        )
        self.assertEqual(self.policy.get_agent_state(state), expected_state)

    def test_is_motor_only_step_returns_false_if_motor_only_step_is_not_in_agent_state(
        self,
    ):
        state = MotorSystemState(
            {
                f"{self.agent_id}": self.default_agent_state,
            }
        )
        self.assertFalse(self.policy.is_motor_only_step(state))

    def test_is_motor_only_step_returns_true_if_motor_only_step_is_true_in_agent_state(
        self,
    ):
        state = MotorSystemState(
            {
                f"{self.agent_id}": {
                    **self.default_agent_state,
                    "motor_only_step": True,
                },
            }
        )
        self.assertTrue(self.policy.is_motor_only_step(state))

    def test_is_motor_only_step_returns_false_if_motor_only_step_is_false_in_agent_state(  # noqa: E501
        self,
    ):
        state = MotorSystemState(
            {
                f"{self.agent_id}": {
                    **self.default_agent_state,
                    "motor_only_step": False,
                },
            }
        )
        self.assertFalse(self.policy.is_motor_only_step(state))


if __name__ == "__main__":
    unittest.main()
