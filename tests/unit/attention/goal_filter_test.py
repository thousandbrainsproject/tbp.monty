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
from unittest.mock import MagicMock

from hypothesis import given

from tbp.monty.attention.goal_filter import NoopGoalFilter
from tbp.monty.cmp import Goal
from tests.strategies.cmp import goals


class NoopGoalFilterTest(unittest.TestCase):
    @given(goals=goals())
    def test_noop_goal_filter_returns_all_goals(self, goals: list[Goal]):
        self.assertEqual(NoopGoalFilter()(MagicMock(), goals), goals)


class HardGoalFilterTest(unittest.TestCase):
    def test_out_of_grid_goals_pass_when_grid_is_empty(self):
        pass

    def test_out_of_grid_goals_pass_when_all_voxel_weights_are_negative(self):
        pass

    def test_out_of_grid_goals_filtered_out_when_there_are_voxels_with_positive_weights(
        self,
    ):  # noqa: E501
        pass

    def test_goals_in_voxels_with_negative_weights_filtered(self):
        pass

    def test_goals_in_voxels_with_positive_weights_pass(self):
        pass
