# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import ClassVar, Protocol, Sequence

import numpy as np

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.memento import Memento


class AttentionSystemProtocol(Protocol):
    def step(
        self, goals: Sequence[Goal], regions: Sequence[AttentionRegion]
    ) -> list[Goal]: ...

    def reset(self) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystem(AttentionSystemProtocol):
    def step(
        self,
        goals: Sequence[Goal],
        regions: Sequence[AttentionRegion],  # noqa: ARG002
    ) -> list[Goal]:
        return list(goals)

    def reset(self) -> None:
        """Nothing to reset."""

    def state_dict(self) -> Memento:
        return {}


class DefaultAttentionSystem(AttentionSystemProtocol):
    MIN_ATTENTION_WEIGHT: ClassVar[float] = -1.0
    """Full inhibition."""
    MAX_ATTENTION_WEIGHT: ClassVar[float] = 1.0
    """Full excitation."""
    WEIGHT_EXPIRATION_TOLERANCE: ClassVar[float] = 1e-6
    """Voxels whose weight magnitude falls below this are expired from the grid."""

    @classmethod
    def expire(cls, grid: VoxelGrid) -> VoxelGrid:
        """Returns the grid removing voxels with weights close enough to zero."""
        data = grid.to_pandas()
        if len(data) == 0:
            return grid
        expiring = data["weight"].abs() < cls.WEIGHT_EXPIRATION_TOLERANCE
        if np.any(expiring):
            return VoxelGrid.from_pandas(grid.voxel_size, data[~expiring])
        return grid

    def step(
        self,
        goals: Sequence[Goal],
        regions: Sequence[AttentionRegion],  # noqa: ARG002
    ) -> list[Goal]:
        return list(goals)

        # proposed_grid = self.voxelize_attention_regions(regions)
        # self._telemetry.proposed_grid(proposed_grid)
        # # Decay what is already held before folding in what was just proposed,
        # # so that a re-proposed voxel's fresh row lands on top of the tick
        # # rather than after it.

        # self._decay(self._grid)
        # self._grid = AttentionSystem.expire(self._grid)
        # self._grid = self._merge(self._grid, proposed_grid)

        # self._telemetry.grid(self._grid)

        # return self._goal_filter(self._grid, goals)

    def reset(self) -> None:
        """Nothing to reset."""

    def state_dict(self) -> Memento:
        return {}
