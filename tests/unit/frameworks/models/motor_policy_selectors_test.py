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
from unittest.mock import Mock

from tbp.monty.frameworks.models.motor_policy_selectors import SinglePolicySelector


class SinglePolicySelectorTest(unittest.TestCase):
    def setUp(self):
        self.policy = Mock()
        self.selector = SinglePolicySelector(self.policy)

    def test_returns_configured_policy(self):
        policy, _ = self.selector([])
        self.assertIs(policy, self.policy)

    def test_returns_none_if_no_goals_given(self):
        _, goal = self.selector([])
        self.assertIsNone(goal)

    def test_returns_goal_with_highest_confidence(self):
        best_goal = Mock(confidence=0.9)
        second_best_goal = Mock(confidence=0.8)
        _, goal = self.selector([second_best_goal, best_goal])
        self.assertIs(goal, best_goal)

    def test_returns_first_goal_with_highest_confidence_if_ties(self):
        first_goal = Mock(confidence=0.9)
        second_goal = Mock(confidence=0.9)
        _, goal = self.selector([first_goal, second_goal])
        self.assertIs(goal, first_goal)

    def test_pre_episode_calls_pre_episode_on_policy(self):
        motor_system = Mock()
        self.selector.pre_episode(motor_system)
        self.policy.pre_episode.assert_called_once_with(motor_system)

    def test_state_dict_returns_state_dict_of_policy(self):
        state_dict = Mock()
        self.policy.state_dict.return_value = state_dict
        self.assertIs(self.selector.state_dict(), state_dict)
