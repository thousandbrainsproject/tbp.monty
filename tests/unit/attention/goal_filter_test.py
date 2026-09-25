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
from tbp.monty.cmp import Goal
from tests.strategies.cmp import goals
from tests.unit.attention.strategies import (
    VoxelGridAndGoals,
    all_negative_default_attention_system_weights,
    default_voxel_grid,
    voxel_grid_and_goals,
    with_negative_default_attention_system_weights,
    with_positive_default_attention_system_weights,
)


class NoopGoalFilterTest(unittest.TestCase):
    @given(goals=goals())
    def test_noop_goal_filter_returns_all_goals(self, goals: list[Goal]) -> None:
        self.assertEqual(NoopGoalFilter()(MagicMock(), goals), goals)


class HardGoalFilterTest(unittest.TestCase):
    @given(voxel_grid=default_voxel_grid(voxels_strategy=st.just([])), goals=goals())
    def test_out_of_grid_goals_pass_when_grid_is_empty(
        self, voxel_grid, goals: list[Goal]
    ) -> None:
        self.assertEqual(HardGoalFilter()(voxel_grid, goals), goals)

    @given(
        grid_and_goals=voxel_grid_and_goals(
            weights_strategy=all_negative_default_attention_system_weights
        )
    )
    def test_out_of_grid_goals_pass_when_all_voxel_weights_are_negative(
        self, grid_and_goals: VoxelGridAndGoals
    ) -> None:
        voxel_grid = grid_and_goals.voxel_grid
        goals = grid_and_goals.goals_out_of_grid
        self.assertEqual(HardGoalFilter()(voxel_grid, goals), goals)

    @given(
        grid_and_goals=voxel_grid_and_goals(
            weights_strategy=with_positive_default_attention_system_weights
        )
    )
    def test_out_of_grid_goals_filtered_out_when_there_are_voxels_with_positive_weights(
        self, grid_and_goals: VoxelGridAndGoals
    ) -> None:
        voxel_grid = grid_and_goals.voxel_grid
        goals = grid_and_goals.goals_out_of_grid
        self.assertEqual(HardGoalFilter()(voxel_grid, goals), [])

    @given(
        grid_and_goals=voxel_grid_and_goals(
            weights_strategy=with_negative_default_attention_system_weights
        )
    )
    def test_goals_in_voxels_with_negative_weights_filtered(
        self, grid_and_goals: VoxelGridAndGoals
    ) -> None:
        voxel_grid = grid_and_goals.voxel_grid
        goals = grid_and_goals.goals_in_negative_weight_grid
        kept = HardGoalFilter()(voxel_grid, goals)
        self.assertEqual(kept, [])

    @given(
        grid_and_goals=voxel_grid_and_goals(
            weights_strategy=with_positive_default_attention_system_weights
        )
    )
    def test_goals_in_voxels_with_positive_weights_pass(
        self, grid_and_goals: VoxelGridAndGoals
    ) -> None:
        voxel_grid = grid_and_goals.voxel_grid
        goals = grid_and_goals.goals_in_positive_weight_grid
        kept = HardGoalFilter()(voxel_grid, goals)
        self.assertEqual(kept, goals)
