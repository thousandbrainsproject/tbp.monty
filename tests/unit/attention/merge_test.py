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
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.attention.merge import Union
from tbp.monty.attention.voxel_grid import VoxelGrid
from tests.unit.attention import strategies


class UnionTest(unittest.TestCase):
    def setUp(self):
        self.union = Union()

    @given(grid_a=strategies.default_voxel_grid())
    def test_raises_value_error_if_voxel_sizes_do_not_match(self, grid_a: VoxelGrid):
        grid_b = VoxelGrid.from_pandas(grid_a.voxel_size + 1, grid_a.to_pandas())
        with self.assertRaisesRegex(ValueError, "Voxel sizes must match for merging."):
            self.union(grid_a, grid_b)

    @given(
        grid_a=strategies.default_voxel_grid(voxel_size_strategy=st.just(1.0)),
        grid_b=strategies.default_voxel_grid(voxel_size_strategy=st.just(1.0)),
        grid_result=strategies.default_voxel_grid(voxel_size_strategy=st.just(1.0)),
    )
    def test_returns_voxel_grid_from_grid_b_combine_first_grid_a(
        self,
        grid_a: VoxelGrid,
        grid_b: VoxelGrid,
        grid_result: VoxelGrid,
    ):
        with patch(
            "pandas.DataFrame.combine_first",
            autospec=True,
            return_value=grid_result.to_pandas(),
        ) as combine_first_mock:
            result = self.union(grid_a, grid_b)

        combine_first_mock.assert_called_once_with(
            grid_b.to_pandas(), grid_a.to_pandas()
        )
        self.assertIs(result.voxel_size, grid_a.voxel_size)
        self.assertIs(result.to_pandas(), grid_result.to_pandas())
