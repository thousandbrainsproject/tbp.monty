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

from tbp.monty.cmp import AttentionRegion
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionProposer,
)

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
        RegionContext,
    )


class InhibitAllOnRecognition(RegionProposer):
    """Signal the attention system to inhibit everything once the LM recognizes.

    The step the LM is down to a single possible match with a unique pose, it
    proposes an empty region carrying the ``inhibit_all`` signal. A merge that
    honors it (``InhibitionFlipsGrid``) flips every voxel the attention system
    holds to full inhibition, dropping the goals it had been attracting so
    the sensor moves on. No locations are proposed, so, unlike
    ``InhibitRecognizedObject``, a poor pose estimate cannot put inhibition
    somewhere it does not belong.

    The signal is sent once per recognized object per episode: recognition
    is an event to react to, not a state to keep re-asserting, and
    re-sending it every step would also inhibit whatever the sensor modules
    propose next.
    """

    def __init__(self) -> None:
        self._signalled: set[str] = set()

    def __call__(self, context: RegionContext) -> AttentionRegion | None:
        """Propose the inhibit-all signal, or nothing.

        Args:
            context: The LM's current region context.

        Returns:
            An empty region carrying the signal the first step an object is
            recognized; None before that and every step after.
        """
        object_id = context.recognized_object
        if object_id is None or object_id in self._signalled:
            return None
        self._signalled.add(object_id)
        return AttentionRegion.empty(inhibit_all=True)

    def reset(self) -> None:
        """Forget which objects were signalled, for the next episode."""
        self._signalled = set()
