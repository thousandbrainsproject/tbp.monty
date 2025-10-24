# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

from tbp.monty.frameworks.models.monty_base import MontyBase


class MontyBasePrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sm1 = MagicMock()
        self.sm1.sensor_module_id = "sm1"
        self.sm2 = MagicMock()
        self.sm2.sensor_module_id = "sm2"
        self.lm1 = MagicMock()
        self.lm2 = MagicMock()
        self.lm3 = MagicMock()
        self.monty_base = MontyBase(
            sensor_modules=[self.sm1, self.sm2],
            learning_modules=[self.lm1, self.lm2, self.lm3],
            motor_system=MagicMock(),
            sm_to_agent_dict={"sm1": "agent_id_0", "sm2": "agent_id_0"},
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=[[], [], []],
            min_eval_steps=10,
            min_train_steps=10,
            num_exploratory_steps=10,
            max_total_steps=100,
        )

    def test_pass_goal_states_collects_all_goals_from_learning_and_sensor_modules(
        self,
    ) -> None:
        self.monty_base.step_type = "matching_step"
        self.lm1.propose_goal_states.return_value = []
        self.lm2.propose_goal_states.return_value = [sentinel.lm2_goal]
        self.lm3.propose_goal_states.return_value = [
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
        ]
        self.sm1.propose_goal_states.return_value = []
        self.sm2.propose_goal_states.return_value = [
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.monty_base._pass_goal_states()

        expected = set(
            {
                sentinel.lm2_goal,
                sentinel.lm3_goal_1,
                sentinel.lm3_goal_2,
                sentinel.sm2_goal_1,
                sentinel.sm2_goal_2,
            }
        )
        self.assertEqual(set(self.monty_base.gsg_outputs), expected)
