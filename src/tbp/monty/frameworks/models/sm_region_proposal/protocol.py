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

    from tbp.monty.cmp import AttentionRegion, Goal, Message
    from tbp.monty.frameworks.models.motor_system_state import SensorState


class SMRegionContext(Protocol):
    """What a sensor module exposes to its region proposers.

    A narrow, read-only view of the module's current step: what it sensed,
    where its sensor is, and the goals it proposed. Proposers never touch
    the module itself.
    """

    @property
    def sensor_module_id(self) -> str:
        """The proposing module's id, for the regions' sender."""

    @property
    def goals(self) -> Sequence[Goal]:
        """The goals the module proposed this step; empty when it proposed none."""

    @property
    def percept(self) -> Message | None:
        """The module's percept this step; None before its first step."""

    @property
    def sensor_state(self) -> SensorState:
        """The sensor's proprioceptive state this step."""


class SMRegionProposer(Protocol):
    """A strategy that turns a sensor module's step into an attention region.

    The sensor module's counterpart of a learning module's RegionProposer:
    called once per Monty step with the module's current context; returns
    the region the module wants folded into the attention system's voxel
    grid this step (negative weights inhibit, positive weights attract),
    None when it has nothing to say.
    """

    def __call__(self, context: SMRegionContext) -> AttentionRegion | None: ...

    def reset(self) -> None:
        """Forget any per-episode state."""
