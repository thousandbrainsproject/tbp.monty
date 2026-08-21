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

from tbp.monty.attention.decay import LinearDecay, NoopDecay
from tbp.monty.attention.voxel_grid import (
    DEFAULT_VOXEL_SIZE,
    VOXEL_LEVELS,
    VoxelGrid,
)
from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, MIN_ATTENTION_WEIGHT

from .strategies import MAX_POINTS, valid_weights


def grid_with_weights(*weights: float) -> VoxelGrid:
    """Build a grid with one voxel per given weight.

    Returns:
        A grid holding voxels (0,0,0), (1,0,0), ... carrying the weights.

    """
    frame = pd.DataFrame(
        {"weight": list(weights)},
        index=pd.MultiIndex.from_tuples(
            [(i, 0, 0) for i in range(len(weights))], names=VOXEL_LEVELS
        ),
    )
    return VoxelGrid(DEFAULT_VOXEL_SIZE, frame)


def voxel_grid_with_weights(weights: np.ndarray) -> VoxelGrid:
    """Build a grid with one voxel per given weight.

    Returns:
        A grid holding voxels (0,0,0), (1,0,0), ... carrying the weights.

    """
    frame = pd.DataFrame(
        {"weight": weights},
        index=pd.MultiIndex.from_tuples(
            [(i, 0, 0) for i in range(len(weights))], names=VOXEL_LEVELS
        ),
    )
    return VoxelGrid(DEFAULT_VOXEL_SIZE, frame)


class NoopDecayTest(unittest.TestCase):
    def test_the_grid_is_left_unchanged(self) -> None:
        n_voxels = 11
        voxels = [(i, 0, 0) for i in range(n_voxels)]
        weights = np.linspace(
            MIN_ATTENTION_WEIGHT,
            MAX_ATTENTION_WEIGHT,
            num=n_voxels,
        )
        df = pd.DataFrame(
            {"weight": weights},
            index=pd.MultiIndex.from_tuples(voxels, names=VOXEL_LEVELS),
        )
        grid = VoxelGrid(DEFAULT_VOXEL_SIZE, df)

        NoopDecay()(grid)

        np.testing.assert_array_equal(grid["weight"].to_numpy(), weights)


MIN_LINEAR_DECAY_RATE = 1e-6
MAX_LINEAR_DECAY_RATE = 10.0

linear_decay_rates = st.floats(
    min_value=MIN_LINEAR_DECAY_RATE, max_value=MAX_LINEAR_DECAY_RATE, allow_nan=False
)
non_positive_decay_rates = st.floats(max_value=0.0)


class LinearDecayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decay = LinearDecay(rate=0.1)

    @given(weights=valid_weights(100), rate=linear_decay_rates)
    def test_moves_weights_toward_zero_by_the_rate_and_clamps_near_zero(
        self,
        weights: np.ndarray,
        rate: float,
    ) -> None:
        # The documented rule: step each weight toward zero by the rate, and a
        # weight that would land within the rate of zero is clamped to zero.
        # A step lands within the rate of zero exactly when |weight| <= 2 * rate,
        # which splits the weights into three disjoint groups.
        grid = voxel_grid_with_weights(weights)
        pre_step_weights = grid.to_pandas()["weight"].to_numpy()

        far_negative = pre_step_weights < -2 * rate
        near_zero = np.abs(pre_step_weights) <= 2 * rate
        far_positive = pre_step_weights > 2 * rate
        assert (far_negative | near_zero | far_positive).all()

        LinearDecay(rate=rate)(grid)

        post_step_weights = grid.to_pandas()["weight"].to_numpy()
        np.testing.assert_allclose(
            post_step_weights[far_negative], pre_step_weights[far_negative] + rate
        )
        np.testing.assert_array_equal(post_step_weights[near_zero], 0.0)
        np.testing.assert_allclose(
            post_step_weights[far_positive], pre_step_weights[far_positive] - rate
        )

    def test_decays_in_place_and_returns_nothing(self) -> None:
        grid = grid_with_weights(3.0)
        frame = grid.to_pandas()
        returned = self.decay(grid)

        self.assertIsNone(returned)
        self.assertIs(grid.to_pandas(), frame)
        np.testing.assert_allclose(frame["weight"].to_numpy(), [2.9])

    @given(weights=valid_weights(MAX_POINTS), rate=non_positive_decay_rates)
    def test_a_non_positive_rate_disables_decay(
        self,
        weights: np.ndarray,
        rate: float,
    ) -> None:
        grid = grid_with_weights(*weights)
        LinearDecay(rate=rate)(grid)

        np.testing.assert_array_equal(grid["weight"].to_numpy(), weights)

    def test_an_empty_grid_stays_empty(self) -> None:
        grid = VoxelGrid(DEFAULT_VOXEL_SIZE)
        self.decay(grid)

        self.assertEqual(len(grid), 0)
