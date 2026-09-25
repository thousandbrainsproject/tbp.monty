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
from unittest.mock import ANY, MagicMock, patch, sentinel

import hypothesis
import pandas as pd
from hypothesis import given

from tbp.monty.attention.attention_system import (
    AttentionRegion,
    DefaultAttentionSystem,
    NoopAttentionSystem,
)
from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import Goal
from tests.strategies.cmp import attention_regions, goals
from tests.unit.attention import strategies


class NoopAttentionSystemTest(unittest.TestCase):
    @given(goals=goals(), regions=attention_regions())
    def test_step_does_not_filter_out_any_goals(
        self, goals: list[Goal], regions: list[AttentionRegion]
    ) -> None:
        system = NoopAttentionSystem()
        filtered_goals = system.step(goals, regions)
        self.assertListEqual(filtered_goals, goals)

    @given(goals=goals(), regions=attention_regions())
    def test_reset_does_nothing(
        self, goals: list[Goal], regions: list[AttentionRegion]
    ) -> None:
        system = NoopAttentionSystem()
        filtered_goals_1 = system.step(goals, regions)
        self.assertListEqual(filtered_goals_1, goals)

        system.reset()
        filtered_goals_2 = system.step(goals, regions)
        self.assertListEqual(filtered_goals_1, filtered_goals_2)

    def test_state_dict_returns_empty_memento(self) -> None:
        system = NoopAttentionSystem()
        memento = system.state_dict()
        self.assertDictEqual(memento, {})


class DefaultAttentionSystemTest(unittest.TestCase):
    @given(grid=strategies.default_voxel_grid())
    def test_expire_removes_voxels_with_weights_below_weight_expiration_tolerance(
        self, grid: VoxelGrid
    ) -> None:
        hypothesis.note(grid.to_pandas()["weight"])
        result = DefaultAttentionSystem.expire(grid)
        self.assertFalse(
            (
                result.to_pandas()["weight"].abs()
                < DefaultAttentionSystem.WEIGHT_EXPIRATION_TOLERANCE
            ).any()
        )

    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_concatenates_regions(
        self, mock_concat: MagicMock
    ) -> None:
        system = DefaultAttentionSystem()
        system._voxelize_attention_regions(sentinel.regions)
        mock_concat.assert_called_once_with(sentinel.regions)

    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_concatenates_regions_returns_empty_voxel_grid_when_concatenated_region_is_empty(  # noqa: E501
        self, mock_concat: MagicMock
    ) -> None:
        mock_region = MagicMock()
        mock_region.__len__.return_value = 0
        mock_concat.return_value = mock_region
        system = DefaultAttentionSystem()

        grid = system._voxelize_attention_regions(sentinel.regions)

        mock_concat.assert_called_once_with(sentinel.regions)
        self.assertEqual(len(grid), 0)

    @given(voxel_grid=strategies.default_voxel_grid())
    @patch("tbp.monty.attention.attention_system.DefaultAttentionSystem._pool_weights")
    @patch("tbp.monty.attention.attention_system.voxelize_and_bin_points")
    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_voxelizes_and_bins_points_when_concatenated_region_is_not_empty(  # noqa: E501
        self,
        mock_concat: MagicMock,
        mock_voxelize_and_bin_points: MagicMock,
        mock_pool_weights: MagicMock,
        voxel_grid: VoxelGrid,
    ) -> None:
        mock_region = MagicMock()
        mock_region.__len__.return_value = 1
        mock_concat.return_value = mock_region
        mock_pool_weights.return_value = voxel_grid.to_pandas()
        system = DefaultAttentionSystem(voxel_size=voxel_grid.voxel_size)

        system._voxelize_attention_regions(sentinel.regions)

        mock_voxelize_and_bin_points.assert_called_once_with(
            system._voxel_size,
            mock_region.locations,
            mock_region.weights,
        )

    @given(voxel_grid=strategies.default_voxel_grid())
    @patch("tbp.monty.attention.attention_system.DefaultAttentionSystem._pool_weights")
    @patch("tbp.monty.attention.attention_system.voxelize_and_bin_points")
    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_pools_weights(
        self,
        mock_concat: MagicMock,
        mock_voxelize_and_bin_points: MagicMock,
        mock_pool_weights: MagicMock,
        voxel_grid: VoxelGrid,
    ) -> None:
        mock_region = MagicMock()
        mock_region.__len__.return_value = 1
        mock_concat.return_value = mock_region
        mock_voxelize_and_bin_points.return_value = sentinel.points
        mock_pool_weights.return_value = voxel_grid.to_pandas()
        system = DefaultAttentionSystem(voxel_size=voxel_grid.voxel_size)

        system._voxelize_attention_regions(sentinel.regions)

        mock_pool_weights.assert_called_once_with(sentinel.points)

    @given(voxel_grid=strategies.default_voxel_grid())
    @patch("tbp.monty.attention.attention_system.DefaultAttentionSystem._pool_weights")
    @patch("tbp.monty.attention.attention_system.voxelize_and_bin_points")
    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_creates_voxel_grid(
        self,
        mock_concat: MagicMock,
        mock_voxelize_and_bin_points: MagicMock,  # noqa: ARG002
        mock_pool_weights: MagicMock,
        voxel_grid: VoxelGrid,
    ) -> None:
        mock_region = MagicMock()
        mock_region.__len__.return_value = 1
        mock_concat.return_value = mock_region
        mock_pool_weights.return_value = voxel_grid.to_pandas()
        system = DefaultAttentionSystem(voxel_size=voxel_grid.voxel_size)

        grid = system._voxelize_attention_regions(sentinel.regions)

        pd.testing.assert_frame_equal(grid.to_pandas(), voxel_grid.to_pandas())
        self.assertEqual(grid.voxel_size, voxel_grid.voxel_size)

    @given(regions=attention_regions())
    @patch(
        "tbp.monty.attention.attention_system.DefaultAttentionSystem._voxelize_attention_regions"
    )
    def test_step_voxelizes_attention_regions(
        self,
        mock_voxelize_attention_regions: MagicMock,
        regions: list[AttentionRegion],
    ) -> None:
        system = DefaultAttentionSystem(merge=MagicMock())
        system.step(MagicMock(), regions)
        mock_voxelize_attention_regions.assert_called_once_with(regions)

    @given(regions=attention_regions())
    @patch(
        "tbp.monty.attention.attention_system.DefaultAttentionSystem._voxelize_attention_regions"
    )
    def test_step_updates_telemetry_with_proposed_grid(
        self,
        mock_voxelize_attention_regions: MagicMock,
        regions: list[AttentionRegion],
    ) -> None:
        mock_telemetry = MagicMock()
        system = DefaultAttentionSystem(merge=MagicMock(), telemetry=mock_telemetry)
        system.step(MagicMock(), regions)
        mock_telemetry.proposed_grid.assert_called_once_with(
            mock_voxelize_attention_regions.return_value
        )

    @given(regions=attention_regions())
    def test_step_decays_current_grid(
        self,
        regions: list[AttentionRegion],
    ) -> None:
        mock_decay = MagicMock()
        system = DefaultAttentionSystem(decay=mock_decay)
        current_grid = system._grid

        system.step(MagicMock(), regions)

        mock_decay.assert_called_once_with(current_grid)

    @given(regions=attention_regions())
    @patch("tbp.monty.attention.attention_system.DefaultAttentionSystem.expire")
    def test_step_expires_current_grid(
        self, mock_expire: MagicMock, regions: list[AttentionRegion]
    ) -> None:
        mock_merge = MagicMock()
        system = DefaultAttentionSystem(merge=mock_merge)
        current_grid = system._grid

        system.step(MagicMock(), regions)

        mock_expire.assert_called_once_with(current_grid)
        mock_merge.assert_called_once_with(mock_expire.return_value, ANY)

    @given(regions=attention_regions())
    @patch("tbp.monty.attention.attention_system.DefaultAttentionSystem.expire")
    @patch(
        "tbp.monty.attention.attention_system.DefaultAttentionSystem._voxelize_attention_regions"
    )
    def test_step_merges_current_grid_with_proposed_grid(
        self,
        mock_voxelize_attention_regions: MagicMock,
        mock_expire: MagicMock,
        regions: list[AttentionRegion],
    ) -> None:
        mock_merge = MagicMock()
        system = DefaultAttentionSystem(merge=mock_merge)
        mock_voxelize_attention_regions.return_value = sentinel.proposed_grid
        mock_expire.return_value = sentinel.current_grid

        system.step(MagicMock(), regions)

        mock_merge.assert_called_once_with(
            sentinel.current_grid, sentinel.proposed_grid
        )
        self.assertEqual(system._grid, mock_merge.return_value)

    def test_step_updates_telemetry_with_merged_grid(self) -> None:
        mock_merge = MagicMock()
        mock_telemetry = MagicMock()
        system = DefaultAttentionSystem(merge=mock_merge, telemetry=mock_telemetry)

        system.step(MagicMock(), MagicMock())

        mock_telemetry.grid.assert_called_once_with(mock_merge.return_value)

    @given(goals=goals(), regions=attention_regions())
    def test_step_returns_filtered_goals(
        self, goals: list[Goal], regions: list[AttentionRegion]
    ) -> None:
        mock_merge = MagicMock()
        mock_merge.return_value = sentinel.grid
        mock_goal_filter = MagicMock()
        system = DefaultAttentionSystem(merge=mock_merge, goal_filter=mock_goal_filter)

        result = system.step(goals, regions)

        mock_goal_filter.assert_called_once_with(sentinel.grid, goals)
        self.assertEqual(result, mock_goal_filter.return_value)

    @given(grid=strategies.default_voxel_grid())
    def test_reset_sets_grid_to_empty(self, grid: VoxelGrid) -> None:
        system = DefaultAttentionSystem(voxel_size=grid.voxel_size)
        system._grid = grid
        system.reset()
        self.assertEqual(len(system._grid), 0)

    def test_reset_resets_telemetry(self) -> None:
        mock_telemetry = MagicMock()
        system = DefaultAttentionSystem(telemetry=mock_telemetry)
        system.reset()
        mock_telemetry.reset.assert_called_once()

    def test_state_dict_returns_current_grid(self) -> None:
        system = DefaultAttentionSystem()
        system._grid = MagicMock()
        state = system.state_dict()
        self.assertIn("grid", state)
        self.assertEqual(state["grid"], system._grid)

    def test_state_dict_returns_telemetry_state_dict(self) -> None:
        mock_telemetry = MagicMock()
        system = DefaultAttentionSystem(telemetry=mock_telemetry)
        state = system.state_dict()
        self.assertIn("telemetry", state)
        self.assertEqual(state["telemetry"], mock_telemetry.state_dict.return_value)
