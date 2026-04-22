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

from tbp.monty.frameworks.models.motor_system import MotorSystem


class MotorSystemTest(unittest.TestCase):
    def setUp(self):
        self.policy_selector = Mock()
        self.motor_system = MotorSystem(self.policy_selector)

    def test_pre_episode_calls_pre_episode_on_policy_selector(self):
        self.motor_system.pre_episode()
        self.policy_selector.pre_episode.assert_called_once_with(self.motor_system)

    def test_state_dict_returns_state_dict_of_policy(self):
        state_dict = Mock()
        self.policy_selector.state_dict.return_value = state_dict
        self.assertIs(self.motor_system.state_dict(), state_dict)
