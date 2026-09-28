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

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.attention.decay import LinearWeightDecay, NoopDecay
from tbp.monty.attention.voxel_grid import VoxelGrid
from tests.unit.attention import strategies


class NoopDecayTest(unittest.TestCase):
    @given(grid=strategies.default_voxel_grid())
    def test_the_grid_is_left_unchanged(self, grid: VoxelGrid) -> None:
        before = grid.to_pandas().copy()
        weights_before = before["weight"].to_numpy()

        NoopDecay()(grid)

        after = grid.to_pandas()
        weights_after = after["weight"].to_numpy()

        np.testing.assert_array_equal(weights_after, weights_before)
        pd.testing.assert_frame_equal(after, before)


MIN_LINEAR_WEIGHT_DECAY_RATE = 1e-6
MAX_LINEAR_WEIGHT_DECAY_RATE = 10.0


class LinearWeightDecayTest(unittest.TestCase):
    @given(
        rate=st.floats(
            min_value=-MAX_LINEAR_WEIGHT_DECAY_RATE,
            max_value=0.0,
            allow_nan=False,
            exclude_max=True,
        )
    )
    def test_raises_value_error_for_negative_rate(self, rate: float) -> None:
        with self.assertRaises(ValueError):
            LinearWeightDecay(rate=rate)

    @given(
        grid=strategies.default_voxel_grid(),
        rate=st.floats(
            min_value=MIN_LINEAR_WEIGHT_DECAY_RATE,
            max_value=MAX_LINEAR_WEIGHT_DECAY_RATE,
            allow_nan=False,
        ),
    )
    def test_moves_weights_toward_zero_by_the_rate_and_clamps_near_zero(
        self,
        grid: VoxelGrid,
        rate: float,
    ) -> None:
        weights = grid.to_pandas()["weight"].to_numpy()
        pre_step_weights = weights.copy()
        far_negative = pre_step_weights < -rate
        near_zero = np.abs(pre_step_weights) <= rate
        far_positive = pre_step_weights > rate

        LinearWeightDecay(rate=rate)(grid)

        post_step_weights = weights
        np.testing.assert_allclose(
            post_step_weights[far_negative], pre_step_weights[far_negative] + rate
        )
        np.testing.assert_array_equal(post_step_weights[near_zero], 0.0)
        np.testing.assert_allclose(
            post_step_weights[far_positive], pre_step_weights[far_positive] - rate
        )
