# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np

from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, AttentionRegion
from tbp.monty.frameworks.models.evidence_matching.region_proposal.excite_goal_locations import (  # noqa: E501
    DEFAULT_LATTICE_STEPS,
    DEFAULT_RADIUS,
    sample_ball,
)
from tbp.monty.frameworks.models.sm_region_proposal.protocol import (
    SMRegionContext,
    SMRegionProposer,
)


class ExciteGoalLocations(SMRegionProposer):
    """Excite a ball around each location the sensor module's goals target.

    The sensor module's counterpart of the learning module proposer of the
    same name: a goal the module sends out is a place it wants the sensor to
    go, so the region around that place is proposed at an excitatory weight
    and the attention system holds attention there rather than filtering out
    the very goal the module cares about. Goals without a location are
    skipped.
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

    def __call__(self, context: SMRegionContext) -> AttentionRegion | None:
        """Propose a ball around each of this step's goal locations, or nothing.

        Args:
            context: The sensor module's current region context.

        Returns:
            The balls at the configured weight, sent by the module, or None
            when no goal this step carries a location.
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
        return AttentionRegion.uniform(
            balls, self._weight, sender_id=context.sensor_module_id
        )

    def reset(self) -> None:
        """Nothing is carried between episodes."""
