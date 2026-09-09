# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from tbp.monty.attention.decay import LinearDecay, VoxelGridDecay
from tbp.monty.attention.goal_filter import GoalFilter, HardGoalFilter
from tbp.monty.attention.merge import Union, VoxelGridMerge
from tbp.monty.attention.telemetry import (
    AttentionSystemTelemetry,
    AttentionSystemTelemetryProtocol,
)
from tbp.monty.attention.voxel_grid import (
    DEFAULT_VOXEL_SIZE,
    VOXEL_LEVELS,
    VoxelGrid,
    voxelize_and_bin_points,
)
from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.memento import Memento

# Voxels whose weight magnitude falls below this are expired from the grid.
WEIGHT_EXPIRATION_TOLERANCE = 1e-6


class AttentionSystemProtocol(Protocol):
    def step(
        self, goals: list[Goal], regions: Sequence[AttentionRegion]
    ) -> list[Goal]: ...

    def reset(self) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystem(AttentionSystemProtocol):
    def step(
        self,
        goals: list[Goal],
        regions: Sequence[AttentionRegion],  # noqa: ARG002
    ) -> list[Goal]:
        return list(goals)

    def reset(self) -> None:
        """Nothing to reset."""

    def state_dict(self) -> Memento:
        return {}


class AttentionSystem(AttentionSystemProtocol):
    """Persisteng, LM and SM informed global attention space.

    Each step, sensor and learning modules propose regions in space as a set of
    locations. Those locations are voxelized into voxels which are used to update
    a persistent voxel grid.

    At present, the voxel grid is used to filter out goals that do not fall within
    the voxel grid.
    """

    def __init__(
        self,
        voxel_size: float = DEFAULT_VOXEL_SIZE,
        pool_weights: Callable[[Sequence[float]], float] | None = None,
        decay: VoxelGridDecay | None = None,
        merge: VoxelGridMerge | None = None,
        goal_filter: GoalFilter | None = None,
        telemetry: AttentionSystemTelemetryProtocol | None = None,
    ):
        """Initialize the attention system.

        Args:
            voxel_size: Edge length of a voxel, in meters.
            pool_weights: How to combine the weights of points sharing a
                voxel into that voxel's weight. Defaults to
                ``negative_priority_max_pool`` (max, unless any weight is
                negative, in which case inhibition dominates).
            decay: How voxel weights move toward zero each step they are not
                re-proposed; applied to the grid in place. Defaults to
                ``LinearDecay``.
            merge: How this step's proposed voxels fold into the remembered
                grid. Defaults to ``Union``.
            goal_filter: How goals are filtered against the voxel grid.
                Defaults to ``HardGoalFilter``.
            telemetry: Telemetry storage for the attention system.

        """
        voxel_size = float(voxel_size)
        assert voxel_size > 0, "voxel_size must be positive"
        self._voxel_size = voxel_size
        self._grid = VoxelGrid(voxel_size)
        self._pool_weights = (
            negative_priority_max_pool if pool_weights is None else pool_weights
        )
        self._decay = LinearDecay() if decay is None else decay
        self._merge = Union() if merge is None else merge
        self._goal_filter = HardGoalFilter() if goal_filter is None else goal_filter
        self._telemetry = AttentionSystemTelemetry() if telemetry is None else telemetry

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel, in meters."""
        return self._voxel_size

    @property
    def grid(self) -> VoxelGrid:
        """The persistent voxel grid."""
        return self._grid

    def step(self, goals: list[Goal], regions: Sequence[AttentionRegion]) -> list[Goal]:
        """Update the attention system with new regions and filter goals.

        Args:
            goals: SM- and LM-derived goals to filter.
            regions: SM- and LM-derived regions which are used to update the
                attention system's persistent voxel grid.

        Returns:
            Filtered list of goals.
        """
        proposed_grid = self.voxelize_attention_regions(regions)
        self._telemetry.proposed_grid(proposed_grid)
        # Decay what is already held before folding in what was just proposed,
        # so that a re-proposed voxel's fresh row lands on top of the tick
        # rather than after it.
        self._decay(self._grid)
        merged = self._merge(self._grid, proposed_grid)
        self._grid = self.expire(merged)
        self._telemetry.grid(self._grid)

        return self.filter_goals(goals)

    def reset(self) -> None:
        """Discard the current grid and recorded telemetry."""
        self._grid = VoxelGrid(self._voxel_size)
        self._telemetry.reset()

    def state_dict(self) -> Memento:
        # voxel_size rides along so consumers can convert the exported voxel
        # indices back to world coordinates.
        return dict(
            voxel_size=self._voxel_size,
            **self._telemetry.state_dict(),
        )

    def voxelize_attention_regions(
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
            return VoxelGrid(self._voxel_size, inhibit_all=region.inhibit_all)

        points = voxelize_and_bin_points(
            region.locations, self._voxel_size, features={"weight": region.weights}
        )
        # Pool the weights of the points sharing a voxel into that voxel's weight.
        voxel_weights = points.groupby("voxel")["weight"].agg(self._pool_weights)

        df = pd.DataFrame(
            {"weight": voxel_weights.to_numpy()},
            index=pd.MultiIndex.from_tuples(voxel_weights.index, names=VOXEL_LEVELS),
        )
        return VoxelGrid(
            self._voxel_size, df.sort_index(), inhibit_all=region.inhibit_all
        )

    def expire(self, merged: VoxelGrid) -> VoxelGrid:
        """Drop voxels whose weight has decayed to zero.

        Args:
            merged: The merged grid, possibly holding voxels decayed to zero.

        Returns:
            The grid with expired voxels (weight within
            WEIGHT_EXPIRATION_TOLERANCE of zero) removed.

        """
        data = merged.to_pandas()
        if len(data) == 0:
            return merged
        expiring = data["weight"].abs() < WEIGHT_EXPIRATION_TOLERANCE
        if np.any(expiring):
            return VoxelGrid(self._voxel_size, data[~expiring])
        return merged

    def filter_goals(self, goals: Sequence[Goal]) -> list[Goal]:
        """Filter the goals against the updated grid.

        Every input goal is stamped with ``info["passed_attention_filter"]``
        so one logged goal list carries the filter's decision; a goal without
        that key never passed through an attention system.

        Args:
            goals: The goals collected from all modules this step.

        Returns:
            The goals the configured goal filter kept.

        """
        for g in goals:
            g.info["passed_attention_filter"] = False
        filtered = self._goal_filter(self._grid, goals)
        for g in filtered:
            g.info["passed_attention_filter"] = True
        return filtered


# --------------------------------------------------------------------------------------
# Feature-pooling functions.


def max_pool(values: Sequence[float]) -> float:
    return max(values)


def mean_pool(values: Sequence[float]) -> float:
    return float(np.mean(values))


def negative_priority_max_pool(values: Sequence[float]) -> float:
    """Inhibition-dominating weight pooler.

    Args:
        values: A sequence of float values.

    Returns:
        The maximum value if all are non-negative, otherwise the most negative value.
    """
    return min(values) if any(v < 0 for v in values) else max(values)
