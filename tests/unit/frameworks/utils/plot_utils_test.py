# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from numpy import array

from tbp.monty.frameworks.actions.actions import MoveTangentially
from tbp.monty.frameworks.utils.plot_utils import get_action_name


class GetActionNameTest(unittest.TestCase):
    def test_is_match_step_on_object(self):
        name = get_action_name(
            action_stats=None, step=0, is_match_step=True, obs_on_object=True
        )
        self.assertEqual(name, "updating possible matches")

    def test_is_match_step_not_on_object(self):
        name = get_action_name(
            action_stats=None, step=0, is_match_step=True, obs_on_object=False
        )
        self.assertEqual(name, "patch not on object")

    def test_not_match_step_step_0(self):
        name = get_action_name(
            action_stats=None, step=0, is_match_step=False, obs_on_object=False
        )
        self.assertEqual(name, "not moved yet")

    def test_not_match_step_action_none(self):
        name = get_action_name(
            action_stats=[[None, None, {}]],
            step=1,
            is_match_step=False,
            obs_on_object=False,
        )
        self.assertEqual(name, "None")

    def test_not_match_step_action_name(self):
        action = MoveTangentially(
            agent_id="agent_id_0", distance=13, direction=array([1, 2, 3])
        )
        name = get_action_name(
            action_stats=[[action, {}]],
            step=1,
            is_match_step=False,
            obs_on_object=False,
        )
        self.assertEqual(name, "move_tangentially - distance:13,direction:[1, 2, 3]")
