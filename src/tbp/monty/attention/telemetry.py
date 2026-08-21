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
from tbp.monty.cmp import Goal
from tbp.monty.memento import Memento

__all__ = [
    "AttentionSystemTelemetry",
    "AttentionSystemTelemetryProtocol",
    "NoopAttentionSystemTelemetry",
]


class AttentionSystemTelemetryProtocol(Protocol):
    def reset(self) -> None: ...

    def voxel_grid(self, grid: VoxelGrid) -> None: ...

    def goal_filtering(self, pre: list[Goal], post: list[Goal]) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    def reset(self) -> None:
        pass

    def voxel_grid(self, grid: VoxelGrid) -> None:
        pass

    def goal_filtering(self, pre: list[Goal], post: list[Goal]) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(voxel_grids=[], pre_filter_goals=[], post_filter_goals=[])


class AttentionSystemTelemetry(AttentionSystemTelemetryProtocol):
    def __init__(self) -> None:
        self.voxel_grids: list[VoxelGrid] = []
        self.pre_filter_goals: list[list[Goal]] = []
        self.post_filter_goals: list[list[Goal]] = []

    def reset(self) -> None:
        self.voxel_grids = []
        self.pre_filter_goals = []
        self.post_filter_goals = []

    def voxel_grid(self, grid: VoxelGrid) -> None:
        # Snapshot the backing frame so later steps cannot alter the record.
        self.voxel_grids.append(VoxelGrid(grid.voxel_size, grid.to_pandas().copy()))

    def goal_filtering(self, pre: list[Goal], post: list[Goal]) -> None:
        self.pre_filter_goals.append(list(pre))
        self.post_filter_goals.append(list(post))

    def state_dict(self) -> Memento:
        # The grids ride along as objects; BufferEncoder flattens them at
        # serialization time (see voxel_grid.encode_voxel_grid).
        return dict(
            voxel_grids=list(self.voxel_grids),
            pre_filter_goals=self.pre_filter_goals,
            post_filter_goals=self.post_filter_goals,
        )
