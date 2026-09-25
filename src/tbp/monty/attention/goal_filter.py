# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import Goal


class GoalFilter(Protocol):
    def __call__(self, voxel_grid: VoxelGrid, goals: Sequence[Goal]) -> list[Goal]: ...


class NoopGoalFilter(GoalFilter):
    """Pass every goal through unchanged."""

    def __call__(
        self,
        voxel_grid: VoxelGrid,  # noqa: ARG002
        goals: Sequence[Goal],
    ) -> list[Goal]:
        """Return the goals unchanged.

        Args:
            voxel_grid: Unused.
            goals: The goals to (not) filter.

        Returns:
            The goals, unfiltered.
        """
        return list(goals)


class HardGoalFilter(GoalFilter):
    """Keep only the goals that fall within a positively weighted voxel.

    Rules:
        - If the attention grid is empty, let all goals through. Otherwise...
        - If all voxels are inhibited (negative-weighted), allow all goals to pass
            that do not land in negative voxels.
        - If a goal does not fall within a voxel, filter it out. Otherwise...
        - If the goals' voxel weight is <= 0, filter the goal out. Otherwise,
            let it through.
    """

    def __call__(self, voxel_grid: VoxelGrid, goals: Sequence[Goal]) -> list[Goal]:
        """Filter the goals against the grid.

        Args:
            voxel_grid: The current voxel grid.
            goals: The goals to filter.

        Returns:
            The goals filtered according to the rules.
        """
        if len(voxel_grid) == 0 or len(goals) == 0:
            return list(goals)

        located = [g for g in goals if g.location is not None]
        unlocated = [g for g in goals if g.location is None]
        if not located:
            return unlocated

        points = np.array([g.location for g in located])
        voxel_weights = voxel_grid.weights_at_points(points, fill_value=np.nan)

        all_inhibited = bool((voxel_grid.weights() < 0).all())

        kept = [
            goal
            for goal, voxel_weight in zip(located, voxel_weights)
            if voxel_weight > 0 or (all_inhibited and np.isnan(voxel_weight))
        ]
        return kept + unlocated
