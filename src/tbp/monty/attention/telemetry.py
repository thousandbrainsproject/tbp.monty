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

    def __init__(self) -> None:
        self.voxel_grids: list[pd.DataFrame] = []

    def reset(self) -> None:
        self.voxel_grids = []

    def voxel_grid(self, grid: pd.DataFrame) -> None:
        self.voxel_grids.append(grid.copy())

    def state_dict(self) -> Memento:
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
