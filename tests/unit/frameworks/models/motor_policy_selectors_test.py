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
        self.ctx = Mock()
        self.observations = Mock()
        self.state = Mock()
        self.percept = Mock()
        self.expected_result = Mock()
        self.policy.return_value = self.expected_result

    def test_delegates_to_configured_policy(self):
        self.selector(self.ctx, self.observations, self.state, self.percept, [])
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            None,
        )

    def test_returns_result_from_policy(self):
        result = self.selector(
            self.ctx, self.observations, self.state, self.percept, []
        )
        self.assertIs(result, self.expected_result)

    def test_calls_policy_with_goal_of_highest_confidence(self):
        best_goal = Mock(confidence=0.9)
        second_best_goal = Mock(confidence=0.8)
        self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            [second_best_goal, best_goal],
        )
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            best_goal,
        )

    def test_calls_policy_with_first_goal_when_confidence_tied(self):
        first_goal = Mock(confidence=0.9)
        second_goal = Mock(confidence=0.9)
        self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            [first_goal, second_goal],
        )
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            first_goal,
        )

    def test_pre_episode_calls_pre_episode_on_policy(self):
        motor_system = Mock()
        self.selector.pre_episode(motor_system)
        self.policy.pre_episode.assert_called_once_with(motor_system)

    def test_state_dict_includes_policy_state_dict(self):
        state_dict = Mock()
        self.policy.state_dict.return_value = state_dict
        self.assertIs(self.selector.state_dict()["policy"], state_dict)
