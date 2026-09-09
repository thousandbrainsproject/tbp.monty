# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, AttentionRegion
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionProposer,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
        RegionContext,
    )

# Radius, in meters, of the excited ball around each goal's location.
DEFAULT_RADIUS = 0.01
# Lattice steps per radius when sampling the ball; the resulting point
# spacing (radius / steps, 2.5 mm at the defaults) stays below the attention
# system's default voxel size so the ball voxelizes without gaps.
DEFAULT_LATTICE_STEPS = 4


def sample_ball(center: npt.NDArray, radius: float, lattice_steps: int) -> npt.NDArray:
    """Sample a solid ball on a cubic lattice.

    Args:
        center: (3,) center of the ball.
        radius: Radius of the ball, in the units of center.
        lattice_steps: Lattice steps per radius; the lattice spans
            [-radius, radius] with 2 * lattice_steps + 1 points per axis.

    Returns:
        The (N, 3) lattice points within radius of the center, the center
        itself among them.
    """
    axis = np.linspace(-radius, radius, 2 * lattice_steps + 1)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    offsets = offsets.reshape(-1, 3)
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius]
    return np.asarray(center, dtype=np.float64) + offsets


class ExciteGoalLocations(RegionProposer):
    """Excite a ball around each location the LM's goals target.

    A goal the LM sends out is a place it wants a sensor to go, so the
    region around that place is proposed at an excitatory weight: the
    attention system then holds attention there rather than filtering out
    the very goals (and follow-ups near them) the LM cares about.

    Goals without a location have nothing to excite and are skipped. This
    proposer only ever sees the LM's own goals; SM-generated goals never
    reach an LM's region proposers.
    """

    def __init__(
        self,
        radius: float = DEFAULT_RADIUS,
        lattice_steps: int = DEFAULT_LATTICE_STEPS,
        weight: float = MAX_ATTENTION_WEIGHT,
    ) -> None:
        """Initialize the proposer.

        Args:
            radius: Radius of the excited ball around each goal's location,
                in meters.
            lattice_steps: Lattice steps per radius used to fill the ball.
            weight: The attention weight given to every point of the ball.
        """
        self._radius = radius
        self._lattice_steps = lattice_steps
        self._weight = weight

    def __call__(self, context: RegionContext) -> AttentionRegion | None:
        """Propose a ball around each of this step's goal locations, or nothing.

        Args:
            context: The LM's current region context.

        Returns:
            The balls at the configured weight, or None when no goal this
            step carries a location.
        """
        locations = [g.location for g in context.goals if g.location is not None]
        if not locations:
            return None
        balls = np.vstack(
            [
                sample_ball(location, self._radius, self._lattice_steps)
                for location in locations
            ]
        )
        return AttentionRegion.uniform(balls, self._weight)

    def reset(self) -> None:
        """Nothing is carried between episodes."""
