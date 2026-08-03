# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from src.tbp.monty.cmp import Goal


class AttentionSystem:
    """Global attention system."""

    def __init__(self, voxel_size: float = 0.05):
        self._voxel_size = voxel_size

    def step(self, goals: list[Goal], regions: list[list[Goal]]) -> list[Goal]:
        """Update the attention system with new goals and regions.

        Args:
            goals: A list of goals to update the attention system with.
            regions: A list of regions, where each region is a list of goals.

        Returns:
            Filtered list of goals.
        """
        raise NotImplementedError
