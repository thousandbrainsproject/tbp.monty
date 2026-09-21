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
from hypothesis import strategies as st

from tbp.monty.attention.goal_filter import HardGoalFilter, NoopGoalFilter
from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import Goal
from tests.strategies.cmp import goals
from tests.unit.attention.strategies import (
    VoxelGridAndGoals,
    all_negative_default_attention_system_weights,
    default_attention_system_weights,
    default_voxel_grid,
    out_of_grid_goals_pass_when_all_voxel_weights_are_negative,
    unique_voxels,
    with_negative_default_attention_system_weights,
    with_positive_default_attention_system_weights,
)


class NoopGoalFilterTest(unittest.TestCase):
    @given(goals=goals())
    def test_noop_goal_filter_returns_all_goals(self, goals: list[Goal]):
        self.assertEqual(NoopGoalFilter()(MagicMock(), goals), goals)


class HardGoalFilterTest(unittest.TestCase):
    @given(voxel_grid=default_voxel_grid(voxels_strategy=st.just([])), goals=goals())
    def test_out_of_grid_goals_pass_when_grid_is_empty(
        self, voxel_grid, goals: list[Goal]
    ):
        self.assertEqual(HardGoalFilter()(voxel_grid, goals), goals)

    @given(
        voxel_grid_and_goals=out_of_grid_goals_pass_when_all_voxel_weights_are_negative()
    )
    def test_out_of_grid_goals_pass_when_all_voxel_weights_are_negative(
        self, voxel_grid_and_goals: VoxelGridAndGoals
    ):
        voxel_grid = voxel_grid_and_goals.voxel_grid
        goals = voxel_grid_and_goals.goals_out_of_grid
        self.assertEqual(HardGoalFilter()(voxel_grid, goals), goals)

    # @given(
    #     voxel_grid=default_voxel_grid(
    #         voxels_strategy=unique_voxels(min_voxels=1),
    #         weights_strategy=with_positive_default_attention_system_weights,
    #     ),
    #     goals=goals(),
    # )
    # def test_out_of_grid_goals_filtered_out_when_there_are_voxels_with_positive_weights(
    #     self,
    # ):
    #     pass

    # @given(
    #     voxel_grid=default_voxel_grid(
    #         voxels_strategy=unique_voxels(min_voxels=1),
    #         weights_strategy=with_negative_default_attention_system_weights,
    #     ),
    #     goals=goals(),
    # )
    # def test_goals_in_voxels_with_negative_weights_filtered(self):
    #     pass

    # @given(
    #     voxel_grid=default_voxel_grid(
    #         voxels_strategy=unique_voxels(min_voxels=1),
    #         weights_strategy=with_positive_default_attention_system_weights,
    #     ),
    #     goals=goals(),
    # )
    # def test_goals_in_voxels_with_positive_weights_pass(self):
    #     pass
