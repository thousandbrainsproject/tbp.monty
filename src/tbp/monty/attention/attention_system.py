# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Global attention over the regions of space proposed by SMs and LMs.

Each step, sensor and learning modules propose regions -- lists of goals whose
locations trace out the surface they are attending to. The attention system
voxelizes those locations into a persistent grid, and only goals that land in an
occupied voxel of the updated grid are passed on toward the motor system.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.attention.voxels import VOXEL_LEVELS, voxelize_and_bin_points
from tbp.monty.cmp import Goal


def empty_voxel_grid() -> pd.DataFrame:
    """Build the frame the attention system holds when no voxels are occupied.

    Returns:
        A frame with no rows, an (x, y, z) MultiIndex, and age/count columns.

    """
    return pd.DataFrame(
        {
            "age": pd.Series(dtype=np.int32),
            "count": pd.Series(dtype=np.int32),
        },
        index=pd.MultiIndex.from_tuples([], names=VOXEL_LEVELS),
    )


class AttentionSystem:
    """Global attention system.

    Maintains a voxelized estimate of the regions of space currently attended to,
    merging in the regions delivered on each step. The grid is a memory rather
    than a snapshot of the latest look: a re-observed voxel is refreshed to a full
    age with its count accumulated, one that was not seen ages by a step, and one
    whose age runs out is dropped.
    """

    def __init__(self, voxel_size: float = 0.05, lifetime: int = 6):
        """Initialize the attention system.

        Args:
            voxel_size: Edge length of a voxel, in world units.
            lifetime: How many steps a voxel survives without being re-observed.

        Raises:
            ValueError: If lifetime is not positive.
        """
        if lifetime < 1:
            raise ValueError(f"lifetime must be >= 1, got {lifetime}")
        self._voxel_size = voxel_size
        self._lifetime = lifetime
        self._voxel_grid = empty_voxel_grid()

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel, in world units."""
        return self._voxel_size

    @property
    def lifetime(self) -> int:
        """How many steps a voxel survives without being re-observed."""
        return self._lifetime

    @property
    def grid(self) -> pd.DataFrame:
        """The voxel grid: (x, y, z) MultiIndex rows with age/count columns."""
        return self._voxel_grid

    def step(self, goals: list[Goal], regions: list[list[Goal]]) -> list[Goal]:
        """Update the attention system with new goals and regions.

        Each region's goal locations are voxelized and merged into the grid,
        which then ages and expires as a whole. Goals are filtered against the
        updated grid: only those whose location falls in an occupied voxel are
        returned. Goals without a location pass through, since they cannot be
        spatially filtered. While the grid is empty -- no module has proposed a
        region yet -- goals pass through unfiltered.

        Args:
            goals: A list of goals to update the attention system with.
            regions: A list of regions, where each region is a list of goals.

        Returns:
            Filtered list of goals.
        """
        observed = self._observe(regions)
        # Age what is already held before folding in what was just seen, so that
        # a re-observed voxel's fresh row lands on top of the tick rather than
        # after it.
        aged = self._age(self._voxel_grid)
        merged = self._merge(aged, observed)
        self._voxel_grid = self._expire(merged)
        return self._filter(goals)

    def contains_points(
        self, points: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.bool_]:
        """Test which locations fall within an occupied voxel.

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
        """Discard the current grid."""
        self._voxel_grid = empty_voxel_grid()

    def _observe(self, regions: list[list[Goal]]) -> pd.DataFrame:
        """Voxelize this step's regions into a fresh grid.

        Every observed voxel starts with a full lifetime. Its count is the number
        of regions that saw it this step, so regions from different modules each
        contribute a sighting; ageing and tallying against what is already held
        happen in _merge.

        Args:
            regions: A list of regions, where each region is a list of goals.

        Returns:
            The grid built from this step's regions alone.

        """
        per_region_voxels = []
        for region in regions:
            locations = [g.location for g in region if g.location is not None]
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
                "age": np.full(len(counts), self._lifetime, dtype=np.int32),
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

    def _merge(self, remembered: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
        """Fold this step's observations into the voxels already held.

        A re-observed voxel is replaced wholesale by its fresh row, so its age
        returns to a full lifetime. Its ``count`` is the exception: sightings
        accumulate, so the fresh tally is added to the one already held. A voxel
        that was not seen this step is carried through untouched, keeping
        whatever age it arrived with.

        Args:
            remembered: The voxels held from earlier steps, already aged.
            observed: The grid built from this step's regions alone.

        Returns:
            The merged frame, before expired voxels are dropped.

        """
        if len(remembered) == 0:
            return observed
        if len(observed) == 0:
            return remembered

        fresh = observed.copy()
        seen_before = fresh.index.intersection(remembered.index)
        if len(seen_before):
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
        """Drop voxels whose age has run out.

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
