# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from tbp.monty.cmp import AttentionRegion, Goal


class RegionContext(Protocol):
    """What a learning module exposes to its region proposers.

    A narrow, read-only view: proposers see the LM's current recognition
    state and can ask for an object's learned surface in body coordinates,
    but never touch the LM, its buffer, or its graph memory directly.
    """

    @property
    def possible_matches(self) -> Sequence[str]:
        """The object ids still consistent with the evidence."""

    @property
    def goals(self) -> Sequence[Goal]:
        """The goals the LM proposed this step; empty when it proposed none."""

    @property
    def recognized_object(self) -> str | None:
        """The object id the LM has recognized with a unique pose, or None."""

    @property
    def current_location(self) -> npt.NDArray | None:
        """The sensor's current location in body coordinates, or None."""

    def surface(self, object_id: str) -> tuple[npt.NDArray, npt.NDArray]:
        """The object's learned surface, in the model's reference frame.

        Args:
            object_id: The object whose graph to read.

        Returns:
            The (N, 3) node locations and their (N, 3) unit surface normals.
        """

    def to_body_frame(self, points: npt.NDArray, object_id: str) -> npt.NDArray:
        """Map model-frame points onto the body frame via the object's MLH pose.

        Args:
            points: (N, 3) points in the object's model frame.
            object_id: The object whose most likely pose to apply.

        Returns:
            The (N, 3) points in body coordinates.
        """


class RegionProposer(Protocol):
    """A strategy that turns an LM's recognition state into an attention region.

    Called once per Monty step with the LM's current context; returns the
    region the LM wants folded into the attention system's voxel grid this
    step (negative weights inhibit, positive weights attract), empty when
    it has nothing to say.
    """

    def __call__(self, context: RegionContext) -> AttentionRegion | None: ...

    def reset(self) -> None:
        """Forget any per-episode state."""
