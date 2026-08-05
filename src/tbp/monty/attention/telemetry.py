# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pandas as pd

from tbp.monty.memento import Memento

__all__ = ["AttentionSystemTelemetry"]


class AttentionSystemTelemetry:
    """Keeps track of the attention system's telemetry.

    Records a snapshot of the voxel grid after each step. The state dict
    flattens each snapshot into plain arrays (voxel coordinates plus the age and
    count columns), so it is JSON-encodable by BufferEncoder with no special
    handling.
    """

    def __init__(self) -> None:
        self.voxel_grids: list[pd.DataFrame] = []

    def reset(self) -> None:
        """Reset the telemetry."""
        self.voxel_grids = []

    def voxel_grid(self, grid: pd.DataFrame) -> None:
        """Record a snapshot of the voxel grid.

        Args:
            grid: The attention system's voxel grid after a step.
        """
        self.voxel_grids.append(grid.copy())

    def state_dict(self) -> Memento:
        """Return the recorded voxel grid snapshots.

        Returns:
            Dictionary containing one entry per step in `voxel_grids`, each
            holding the occupied `voxels` as an (N, 3) array and the aligned
            `age` and `count` arrays.
        """
        return dict(
            voxel_grids=[
                dict(
                    voxels=grid.index.to_frame(index=False).to_numpy(),
                    age=grid["age"].to_numpy(),
                    count=grid["count"].to_numpy(),
                )
                for grid in self.voxel_grids
            ]
        )
