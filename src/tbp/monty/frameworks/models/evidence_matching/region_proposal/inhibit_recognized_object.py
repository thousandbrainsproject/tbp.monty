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

from tbp.monty.cmp import MIN_ATTENTION_WEIGHT, AttentionRegion
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionProposer,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
        RegionContext,
    )

# How far, in meters, the inhibited shell extends to either side of the
# learned surface, so that the voxels the sensor will actually report from
# (slightly above or below the model's nodes) are covered.
DEFAULT_SHELL_THICKNESS = 0.02
# How many layers sample the shell on each side of the surface.
DEFAULT_SHELL_LAYERS = 10


def thicken_surface(
    points: npt.NDArray,
    normals: npt.NDArray,
    thickness: float,
    num_layers: int,
) -> npt.NDArray:
    """Pad a surface into a shell by stepping each point along its normal.

    Args:
        points: (N, 3) surface points.
        normals: (N, 3) unit normals, one per point.
        thickness: Half-width of the shell, in the units of points.
        num_layers: Number of offset layers per side, spread evenly over
            (0, thickness].

    Returns:
        The (N * (1 + 2 * num_layers), 3) shell: the original points first,
        then every point displaced by each signed offset.
    """
    offsets = np.linspace(0.0, thickness, num_layers + 1)[1:]
    signed = np.concatenate([-offsets[::-1], offsets])
    layers = points[:, None, :] + signed[None, :, None] * normals[:, None, :]
    return np.vstack([points, layers.reshape(-1, 3)])


class InhibitRecognizedObject(RegionProposer):
    """Inhibit an object's whole surface once the LM has recognized it.

    Once the LM is down to a single possible match with a unique pose, there
    is nothing left to learn from that object this episode: its learned
    surface, thickened into a shell and placed in the body frame via the most
    likely pose, is proposed at an inhibitory weight so that goals on it are
    filtered out and the sensor moves on.
    """

    def __init__(
        self,
        thickness: float = DEFAULT_SHELL_THICKNESS,
        num_layers: int = DEFAULT_SHELL_LAYERS,
        weight: float = MIN_ATTENTION_WEIGHT,
    ) -> None:
        """Initialize the proposer.

        Args:
            thickness: Half-width of the inhibited shell around the surface, in
                meters.
            num_layers: Offset layers per side used to fill the shell.
            weight: The attention weight given to every point of the shell.
        """
        self._thickness = thickness
        self._num_layers = num_layers
        self._weight = weight

    def __call__(self, context: RegionContext) -> AttentionRegion | None:
        """Propose the recognized object's shell, or nothing.

        Args:
            context: The LM's current region context.

        Returns:
            The shell at the configured weight, or None while no object is
            recognized.
        """
        object_id = context.recognized_object
        if object_id is None or context.current_location is None:
            return None
        points, normals = context.surface(object_id)
        shell = thicken_surface(points, normals, self._thickness, self._num_layers)
        return AttentionRegion.uniform(
            context.to_body_frame(shell, object_id), self._weight
        )

    def reset(self) -> None:
        """Nothing is carried between episodes."""
