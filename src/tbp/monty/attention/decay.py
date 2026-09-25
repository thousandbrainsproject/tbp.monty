# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

import numpy as np

from tbp.monty.attention.voxel_grid import VoxelGrid

DEFAULT_LINEAR_WEIGHT_DECAY_RATE = 0.1
"""Default per-step decay toward zero for linear decay."""


class VoxelGridWeightDecay(Protocol):
    def __call__(self, grid: VoxelGrid) -> None: ...


class NoopDecay(VoxelGridWeightDecay):
    """Leave every voxel weight untouched."""

    def __call__(self, grid: VoxelGrid) -> None:
        """Leave the grid as it is.

        Args:
            grid: The grid to (not) decay.
        """


class LinearWeightDecay(VoxelGridWeightDecay):
    """Move each voxel weight toward zero by a fixed amount.

    Decay is applied in place to the grid's weights. Weights that fall
    within [-rate, +rate] get clamped to zero.
    """

    _rate: float

    def __init__(self, rate: float = DEFAULT_LINEAR_WEIGHT_DECAY_RATE) -> None:
        """Initialize the decay.

        Args:
            rate: How much a weight moves toward zero per step.

        Raises:
            ValueError: If the rate is negative.
        """
        if rate < 0.0:
            raise ValueError(f"Rate must be non-negative, got {rate}")
        self._rate = rate

    def __call__(self, grid: VoxelGrid) -> None:
        """Decay every weight in the grid by one step, in place.

        Args:
            grid: The grid to decay.
        """
        if len(grid) == 0 or self._rate == 0.0:
            return
        data = grid.to_pandas()
        weights = data["weight"].to_numpy()
        weights.flags.writeable = True
        to_step = np.abs(weights) > self._rate

        stepped = weights[to_step] - self._rate * np.sign(weights[to_step])
        weights[to_step] = stepped
        weights[~to_step] = 0.0
        data["weight"] = weights
