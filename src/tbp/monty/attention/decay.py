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

# Default per-step decay toward zero for linear decay.
DEFAULT_LINEAR_DECAY_RATE = 0.1


class VoxelGridDecay(Protocol):
    def __call__(self, grid: VoxelGrid) -> None: ...


class NoopDecay(VoxelGridDecay):
    """Leave every voxel weight untouched."""

    def __call__(self, grid: VoxelGrid) -> None:
        """Leave the grid as it is.

        Args:
            grid: The grid to (not) decay.
        """


class LinearDecay(VoxelGridDecay):
    """Move each voxel weight toward zero by a fixed rate per step.

    Decay is applied in place to the grid's backing frame. A weight that
    would land within the rate of zero is clamped to zero instead: stepping
    past zero would flip the sign and oscillate forever, and a remainder
    smaller than the rate is as good as gone. A non-positive rate disables
    decay.
    """

    def __init__(self, rate: float = DEFAULT_LINEAR_DECAY_RATE) -> None:
        """Initialize the decay.

        Args:
            rate: How much a weight moves toward zero per step.
        """
        self._rate = rate

    def __call__(self, grid: VoxelGrid) -> None:
        """Decay every weight in the grid by one step, in place.

        Args:
            grid: The grid to decay.
        """
        if len(grid) == 0 or self._rate <= 0:
            return
        data = grid.to_pandas()
        weights = data["weight"].to_numpy()
        stepped = weights - self._rate * np.sign(weights)
        data["weight"] = np.where(np.abs(stepped) <= self._rate, 0.0, stepped)
