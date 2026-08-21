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

import pandas as pd

from tbp.monty.attention.voxel_grid import VoxelGrid


def _check_can_merge(grid_a: VoxelGrid, grid_b: VoxelGrid) -> None:
    """Check that two grids are compatible for merging.

    Args:
        grid_a: The grid being merged into.
        grid_b: The grid being merged in.
    """
    assert grid_a.voxel_size == grid_b.voxel_size, "Voxel sizes must match for merging."


class VoxelGridMerge(Protocol):
    def __call__(self, grid_a: VoxelGrid, grid_b: VoxelGrid) -> VoxelGrid: ...


class Union(VoxelGridMerge):
    """Merge grid_b into grid_a, taking the union of their voxels.

    Every voxel occupied in either grid is occupied in the result. Where the
    grids overlap, grid_b's weight wins; making that overlap policy
    configurable is left for later.
    """

    def __call__(
        self,
        grid_a: VoxelGrid,
        grid_b: VoxelGrid,
    ) -> VoxelGrid:
        """Merge grid_b into grid_a.

        Args:
            grid_a: The grid being merged into.
            grid_b: The grid being merged in; its weights win on overlap.

        Returns:
            The merged grid.
        """
        _check_can_merge(grid_a, grid_b)

        a_to_merge: pd.DataFrame = grid_a.to_pandas()
        b_to_merge: pd.DataFrame = grid_b.to_pandas()

        overlap: pd.MultiIndex = grid_a.index.intersection(grid_b.index)
        if len(overlap):
            a_to_merge = a_to_merge.drop(overlap, errors="ignore")

        df = pd.concat([a_to_merge, b_to_merge])
        return VoxelGrid(grid_a.voxel_size, df)
