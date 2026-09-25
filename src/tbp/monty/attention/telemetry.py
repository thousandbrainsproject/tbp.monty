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

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.memento import Memento

__all__ = [
    "AttentionSystemTelemetry",
    "DefaultAttentionSystemTelemetry",
    "NoopAttentionSystemTelemetry",
]


class AttentionSystemTelemetry(Protocol):
    def reset(self) -> None: ...

    def proposed_grid(self, grid: VoxelGrid) -> None: ...

    def grid(self, grid: VoxelGrid) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystemTelemetry(AttentionSystemTelemetry):
    def reset(self) -> None:
        pass

    def proposed_grid(self, grid: VoxelGrid) -> None:
        pass

    def grid(self, grid: VoxelGrid) -> None:
        pass

    def state_dict(self) -> Memento:
        return dict(grids=[], proposed_grids=[])


class DefaultAttentionSystemTelemetry(AttentionSystemTelemetry):
    def __init__(self) -> None:
        self._grids: list[VoxelGrid] = []
        self._proposed_grids: list[VoxelGrid] = []

    def reset(self) -> None:
        self._grids = []
        self._proposed_grids = []

    def grid(self, grid: VoxelGrid) -> None:
        self._grids.append(grid)

    def proposed_grid(self, grid: VoxelGrid) -> None:
        self._proposed_grids.append(grid)

    def state_dict(self) -> Memento:
        return dict(
            grids=list(self._grids),
            proposed_grids=list(self._proposed_grids),
        )
