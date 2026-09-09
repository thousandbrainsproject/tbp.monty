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
from unittest.mock import Mock, sentinel

import numpy as np

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_telemetry import MotorSystemTelemetry


class MotorSystemTest(unittest.TestCase):
    def setUp(self):
        self.policy_selector = Mock()
        self.motor_system = MotorSystem(self.policy_selector)

    def test_reset_calls_reset_on_policy_selector(self):
        self.motor_system.reset()
        self.policy_selector.reset.assert_called_once_with()

    def test_state_dict_holds_the_policy_selectors_state_and_the_telemetry(self):
        self.policy_selector.state_dict.return_value = {"policy": sentinel.policy}

        state_dict = self.motor_system.state_dict()

        self.assertEqual(state_dict["policy"], sentinel.policy)
        self.assertEqual(state_dict["telemetry"], {"goals_in": []})

    def test_records_the_goals_it_receives_in_its_telemetry(self):
        telemetry = MotorSystemTelemetry()
        motor_system = MotorSystem(self.policy_selector, telemetry=telemetry)
        self.policy_selector.return_value = Mock(actions=[], telemetry=None)
        self.policy_selector.state_dict.return_value = {}
        goal = Goal(
            location=np.array([1.0, 2.0, 3.0]),
            morphological_features=None,
            non_morphological_features=None,
            confidence=0.5,
            pass_message=True,
            sender_id="view_finder",
            sender_type="SM",
            process_features_in_lm=False,
            goal_tolerances=None,
        )

        motor_system(Mock(), Mock(), {}, Mock(), [goal])
        motor_system(Mock(), Mock(), {}, Mock(), [])

        goals = motor_system.state_dict()["telemetry"]["goals_in"]
        self.assertEqual(len(goals), 2)
        np.testing.assert_array_equal(goals[0]["locations"], [[1.0, 2.0, 3.0]])
        self.assertEqual(list(goals[0]["sender_ids"]), ["view_finder"])
        self.assertEqual(len(goals[1]["confidences"]), 0)

    def test_reset_resets_the_telemetry(self):
        telemetry = Mock()
        motor_system = MotorSystem(self.policy_selector, telemetry=telemetry)

        motor_system.reset()

        telemetry.reset.assert_called_once_with()

    def test_attempted_goal_is_none_before_any_step(self):
        self.assertIsNone(self.motor_system.attempted_goal)

    def test_exposes_attempted_goal_from_the_policy_result(self):
        goal = Mock()
        self.policy_selector.return_value = Mock(
            actions=[], telemetry=None, attempted_goal=goal
        )

        self.motor_system(Mock(), Mock(), {}, Mock(), [])

        self.assertIs(self.motor_system.attempted_goal, goal)

    def test_attempted_goal_is_cleared_on_the_next_step(self):
        self.policy_selector.return_value = Mock(
            actions=[], telemetry=None, attempted_goal=Mock()
        )
        self.motor_system(Mock(), Mock(), {}, Mock(), [])

        self.policy_selector.return_value = Mock(
            actions=[], telemetry=None, attempted_goal=None
        )
        self.motor_system(Mock(), Mock(), {}, Mock(), [])

        self.assertIsNone(self.motor_system.attempted_goal)
