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

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from tbp.monty.cmp import Goal
    from tbp.monty.frameworks.models.evidence_matching.learning_module import (
        EvidenceGraphLM,
    )


class EvidenceLMRegionContext:
    """The RegionContext an EvidenceGraphLM hands to its region proposers.

    Reads only state the LM has already computed this step: the possible
    matches, the terminal state set by the terminal-condition check, and the
    per-object most likely hypothesis. It never calls the pose check itself,
    which advances the LM's symmetry counter as a side effect.
    """

    def __init__(self, lm: EvidenceGraphLM) -> None:
        """Wrap a learning module.

        Args:
            lm: The learning module to expose.
        """
        self._lm = lm

    @property
    def possible_matches(self) -> Sequence[str]:
        """The object ids still consistent with the evidence."""
        return self._lm.get_possible_matches()

    @property
    def goals(self) -> Sequence[Goal]:
        """The goals the LM proposed this step; empty when it proposed none."""
        return tuple(self._lm.propose_goals())

    @property
    def recognized_object(self) -> str | None:
        """The sole possible match once the LM's terminal state is a match."""
        matches = self.possible_matches
        if self._lm.terminal_state == "match" and len(matches) == 1:
            return matches[0]
        return None

    @property
    def current_location(self) -> npt.NDArray | None:
        """The sensor's current location in body coordinates, or None."""
        return self._lm.buffer.current_location()

    def surface(self, object_id: str) -> tuple[npt.NDArray, npt.NDArray]:
        """The object's learned surface from its first input channel.

        Args:
            object_id: The object whose graph to read.

        Returns:
            The (N, 3) node locations and their (N, 3) surface normals, in the
            model's reference frame.
        """
        memory = self._lm.graph_memory
        channel = memory.get_input_channels_in_graph(object_id)[0]
        locations = np.asarray(memory.get_locations_in_graph(object_id, channel))
        # The first pose vector at each node is its surface normal.
        normals = memory.get_rotation_features_at_all_nodes(object_id, channel)[:, 0, :]
        return locations, np.asarray(normals)

    def to_body_frame(self, points: npt.NDArray, object_id: str) -> npt.NDArray:
        """Map model-frame points onto the body frame via the object's MLH pose.

        The MLH stores the sensor's location in the model frame and the
        rotation that takes body-frame displacements into the model frame, so
        a model-frame point maps to the body frame as
        current_location + rotation.inv().apply(point - mlh_location)
        (the same convention the goal generator uses for its targets).

        Args:
            points: (N, 3) points in the object's model frame.
            object_id: The object whose most likely pose to apply.

        Returns:
            The (N, 3) points in body coordinates.
        """
        mlh = self._lm.get_mlh_for_object(object_id)
        displacements = mlh["rotation"].inv().apply(points - mlh["location"])
        return self.current_location + displacements
