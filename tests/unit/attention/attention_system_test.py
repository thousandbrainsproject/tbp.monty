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
from unittest.mock import MagicMock, patch, sentinel

import hypothesis
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.attention.attention_system import (
    AttentionRegion,
    DefaultAttentionSystem,
    NoopAttentionSystem,
)
from tbp.monty.attention.decay import LinearWeightDecay, NoopDecay
from tbp.monty.attention.voxel_grid import (
    VoxelGrid,
)
from tbp.monty.cmp import Goal
from tests.strategies.cmp import attention_regions, goals
from tests.unit.attention import strategies


class NoopAttentionSystemTest(unittest.TestCase):
    @given(goals=goals(), regions=attention_regions())
    def test_step_does_not_filter_out_any_goals(
        self, goals: list[Goal], regions: list[AttentionRegion]
    ):
        system = NoopAttentionSystem()
        filtered_goals = system.step(goals, regions)
        self.assertListEqual(filtered_goals, goals)

    @given(goals=goals(), regions=attention_regions())
    def test_reset_does_nothing(
        self, goals: list[Goal], regions: list[AttentionRegion]
    ):
        system = NoopAttentionSystem()
        filtered_goals_1 = system.step(goals, regions)
        self.assertListEqual(filtered_goals_1, goals)

        system.reset()
        filtered_goals_2 = system.step(goals, regions)
        self.assertListEqual(filtered_goals_1, filtered_goals_2)

    def test_state_dict_returns_empty_memento(self):
        system = NoopAttentionSystem()
        memento = system.state_dict()
        self.assertDictEqual(memento, {})


class DefaultAttentionSystemTest(unittest.TestCase):
    @given(grid=strategies.default_voxel_grid())
    def test_expire_removes_voxels_with_weights_below_weight_expiration_tolerance(
        self, grid: VoxelGrid
    ):
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
    ):
        system = DefaultAttentionSystem()
        system._voxelize_attention_regions(sentinel.regions)
        mock_concat.assert_called_once_with(sentinel.regions)

    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_concatenates_regions_returns_empty_voxel_grid_when_concatenated_region_is_empty(
        self, mock_concat: MagicMock
    ):
        mock_region = MagicMock()
        mock_region.__len__.return_value = 0
        mock_concat.return_value = mock_region
        system = DefaultAttentionSystem()

        grid = system._voxelize_attention_regions(sentinel.regions)

        mock_concat.assert_called_once_with(sentinel.regions)
        self.assertEqual(len(grid), 0)

    @patch("tbp.monty.attention.attention_system.voxelize_and_bin_points")
    @patch("tbp.monty.cmp.AttentionRegion.concat")
    def test_voxelize_attention_regions_voxelizes_and_bins_points_when_concatenated_region_is_not_empty(
        self,
        mock_concat: MagicMock,
        mock_voxelize_and_bin_points: MagicMock,
    ):
        mock_region = MagicMock()
        mock_region.__len__.return_value = 1
        mock_concat.return_value = mock_region
        system = DefaultAttentionSystem()

        system._voxelize_attention_regions(sentinel.regions)

        mock_voxelize_and_bin_points.assert_called_once_with(
            system._voxel_size,
            mock_region.locations,
            mock_region.weights,
        )

    def test_voxelize_attention_regions_pools_weights(self):
        pass

    def test_voxelize_attention_regions_creates_voxel_grid(self):
        pass
