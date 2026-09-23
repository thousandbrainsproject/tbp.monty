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
import pandas as pd

from tbp.monty.attention.decay import LinearWeightDecay, VoxelGridWeightDecay
from tbp.monty.attention.goal_filter import GoalFilter, HardGoalFilter
from tbp.monty.attention.merge import Union, VoxelGridMerge
from tbp.monty.attention.voxel_grid import (
    VOXEL_LEVELS,
    VoxelGrid,
    voxelize_and_bin_points,
)
from tbp.monty.attention.weight_pooler import WeightPooler, negative_priority_max_pool
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

    _voxel_size: float
    _weight_pooler: WeightPooler
    _decay: VoxelGridWeightDecay
    _merge: VoxelGridMerge
    _goal_filter: GoalFilter
    _grid: VoxelGrid

    def __init__(
        self,
        voxel_size: float = 0.05,
        weight_pooler: WeightPooler = negative_priority_max_pool,
        decay: VoxelGridWeightDecay | None = None,
        merge: VoxelGridMerge | None = None,
        goal_filter: GoalFilter | None = None,
    ) -> None:
        self._voxel_size = voxel_size
        self._weight_pooler = weight_pooler
        self._decay = LinearWeightDecay() if decay is None else decay
        self._merge = Union() if merge is None else merge
        self._goal_filter = HardGoalFilter() if goal_filter is None else goal_filter
        self._grid = VoxelGrid.empty(voxel_size)

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
        regions: Sequence[AttentionRegion],
    ) -> list[Goal]:
        proposed_grid = self._voxelize_attention_regions(regions)
        self._decay(self._grid)
        self._grid = DefaultAttentionSystem.expire(self._grid)
        self._grid = self._merge(self._grid, proposed_grid)
        return self._goal_filter(self._grid, goals)

    def reset(self) -> None:
        """Nothing to reset."""

    def state_dict(self) -> Memento:
        return {}

    def _voxelize_attention_regions(
        self, regions: Sequence[AttentionRegion]
    ) -> VoxelGrid:
        """Voxelize this step's regions into a fresh grid.

        Args:
            regions: The regions proposed this step, one per module.

        Returns:
            The grid built from this step's regions alone, carrying the
            inhibit-all signal if any region does.

        """
        region = AttentionRegion.concat(regions)
        if len(region) == 0:
            return VoxelGrid.empty(self._voxel_size)

        points = voxelize_and_bin_points(
            self._voxel_size,
            region.locations,
            region.weights,
        )

        df = self._pool_weights(points)

        return VoxelGrid.from_pandas(self._voxel_size, df)

    def _pool_weights(self, points: pd.DataFrame) -> pd.DataFrame:
        weights = points.groupby("voxel")["weight"].agg(self._weight_pooler)
        return pd.DataFrame(
            {"weight": weights.to_numpy()},
            index=pd.MultiIndex.from_tuples(weights.index, names=VOXEL_LEVELS),
        )
