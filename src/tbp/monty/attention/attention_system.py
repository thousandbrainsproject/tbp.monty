# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.attention.telemetry import AttentionSystemTelemetry
from tbp.monty.attention.voxels import VOXEL_LEVELS, voxelize_and_bin_points
from tbp.monty.cmp import AttentionWeight, Goal
from tbp.monty.memento import Memento


def empty_voxel_grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": pd.Series(dtype=np.int32),
            "count": pd.Series(dtype=np.int32),
        },
        index=pd.MultiIndex.from_tuples([], names=VOXEL_LEVELS),
    )


class AttentionSystem:
    """Persisteng, LM and SM informed global attention space.

    Each step, sensor and learning modules propose regions in space as a set of
    locations. Those locations are voxelized into voxels which are used to update
    a persistent voxel grid.

    At present, the voxel grid is used to filter out goals that do not fall within
    the voxel grid. Voxels that have not been re-proposed for a number of steps
    (i.e., the voxel_lifetime) are expired from the grid.
    """

    def __init__(
        self,
        voxel_size: float = 0.005,
        voxel_lifetime: int = 6,
        telemetry: AttentionSystemTelemetry | None = None,
    ):
        """Initialize the attention system.

        Args:
            voxel_size: Edge length of a voxel, in meters.
            voxel_lifetime: How many steps a voxel survives without being
                re-proposed.
            telemetry: Telemetry storage for the attention system.

        Raises:
            ValueError: If voxel_lifetime is not positive.
        """
        if voxel_lifetime < 1:
            raise ValueError(f"voxel_lifetime must be >= 1, got {voxel_lifetime}")
        self._voxel_size = voxel_size
        self._voxel_lifetime = voxel_lifetime
        self._telemetry = (
            AttentionSystemTelemetry() if telemetry is None else telemetry
        )
        self._voxel_grid = empty_voxel_grid()

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel, in meters."""
        return self._voxel_size

    @property
    def voxel_lifetime(self) -> int:
        """How many steps a voxel survives without being re-proposed."""
        return self._voxel_lifetime

    @property
    def grid(self) -> pd.DataFrame:
        """The voxel grid: (x, y, z) MultiIndex rows with age/count columns."""
        return self._voxel_grid

    def step(
        self, goals: list[Goal], regions: list[list[AttentionWeight]]
    ) -> list[Goal]:
        """Update the attention system with new regions and filter goals.

        Args:
            goals: SM- and LM-derived goals to filter.
            regions: SM- and LM-derived regions which are used to update the
                attention system's persistent voxel grid.

        Returns:
            Filtered list of goals.
        """
        proposed = self._voxelize_regions(regions)
        # Age what is already held before folding in what was just proposed, so
        # that a re-proposed voxel's fresh row lands on top of the tick rather
        # than after it.
        aged = self._age(self._voxel_grid)
        merged = self._merge(aged, proposed)
        self._voxel_grid = self._expire(merged)
        self._telemetry.voxel_grid(self._voxel_grid)
        filtered = self._filter(goals)
        self._telemetry.goal_filtering(goals, filtered)
        return filtered

    def contains_points(
        self,
        points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.bool_]:
        """Test which locations fall within the voxel grid.

        Args:
            points: a (N, 3) array of points.

        Returns:
            A boolean array with shape (N,).

        """
        occupied = self._voxel_grid.index
        points = np.atleast_2d(points)
        if len(occupied) == 0:
            return np.zeros(len(points), dtype=bool)

        indices = np.floor(points / self._voxel_size).astype(int)
        query = pd.MultiIndex.from_arrays(indices.T, names=VOXEL_LEVELS)
        return query.isin(occupied)

    def reset(self) -> None:
        """Discard the current grid and recorded telemetry."""
        self._voxel_grid = empty_voxel_grid()
        self._telemetry.reset()

    def state_dict(self) -> Memento:
        return dict(
            voxel_size=self._voxel_size,
            voxel_lifetime=self._voxel_lifetime,
            **self._telemetry.state_dict(),
        )

    def _voxelize_regions(self, regions: list[list[AttentionWeight]]) -> pd.DataFrame:
        """Voxelize this step's regions into a fresh grid.

        Args:
            regions: A list of regions, where each region is a list of goals.

        Returns:
            The grid built from this step's regions alone.

        """
        per_region_voxels = []
        for region in regions:
            locations = [aw.location for aw in region if aw.location is not None]
            if not locations:
                continue
            per_region_voxels.extend(
                voxelize_and_bin_points(np.asarray(locations), self._voxel_size)
            )
        if not per_region_voxels:
            return empty_voxel_grid()

        index = pd.MultiIndex.from_tuples(per_region_voxels, names=VOXEL_LEVELS)
        counts = (
            pd.Series(1, index=index, dtype=np.int32)
            .groupby(level=list(VOXEL_LEVELS))
            .sum()
            .astype(np.int32)
        )
        return pd.DataFrame(
            {
                "age": np.full(len(counts), self._voxel_lifetime, dtype=np.int32),
                "count": counts,
            }
        )

    def _age(self, remembered: pd.DataFrame) -> pd.DataFrame:
        """Tick every held voxel one step closer to expiring.

        Args:
            remembered: The voxels held going into this step.

        Returns:
            The frame with every age decremented by one.

        """
        if len(remembered) == 0:
            return remembered

        aged = remembered.copy()
        # Subtracting through the frame would widen the dtype, so write back the
        # declared one: age is meant to stay an integer count of steps.
        aged["age"] = (aged["age"] - 1).astype(np.int32)
        return aged

    def _merge(self, remembered: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
        """Merge this step's proposed voxels into the voxels already held.

        Args:
            remembered: The voxels held from earlier steps, already aged.
            proposed: The grid built from this step's regions alone.

        Returns:
            The merged frame, before expired voxels are dropped.

        """
        if len(remembered) == 0:
            return proposed
        if len(proposed) == 0:
            return remembered

        fresh = proposed.copy()
        seen_before = fresh.index.intersection(remembered.index)
        if len(seen_before):
            fresh.loc[seen_before, "age"] = self._voxel_lifetime
            fresh.loc[seen_before, "count"] = (
                fresh.loc[seen_before, "count"].to_numpy()
                + remembered.loc[seen_before, "count"].to_numpy()
            ).astype(np.int32)

        # The fresh row wins outright, so drop the stale one it replaces.
        carried = remembered.drop(index=fresh.index, errors="ignore")
        if len(carried) == 0:
            return fresh
        return pd.concat([fresh, carried])

    @staticmethod
    def _expire(data: pd.DataFrame) -> pd.DataFrame:
        """Drop voxels that haven't been seen in a while.

        Args:
            data: A merged frame, possibly holding voxels aged past their end.

        Returns:
            The frame with expired rows removed.

        """
        if len(data) == 0:
            return data
        return data[data["age"].to_numpy() > 0]

    def _filter(self, goals: Sequence[Goal]) -> list[Goal]:
        """Keep the goals that live in the updated grid.

        Args:
            goals: The goals collected from all modules this step.

        Returns:
            The goals inside an occupied voxel, plus any without a location.
            All goals, if the grid is empty.

        """
        if len(self._voxel_grid) == 0:
            return list(goals)

        located = [g for g in goals if g.location is not None]
        unlocated = [g for g in goals if g.location is None]
        if not located:
            return unlocated

        contained = self.contains_points(
            np.asarray([g.location for g in located])
        )
        return [g for g, keep in zip(located, contained) if keep] + unlocated
