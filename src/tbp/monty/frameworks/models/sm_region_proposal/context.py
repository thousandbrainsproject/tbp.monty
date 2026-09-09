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

from tbp.monty.frameworks.models.sm_region_proposal.protocol import SMRegionContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tbp.monty.cmp import Goal, Message
    from tbp.monty.frameworks.models.motor_system_state import SensorState


class CameraSMRegionContext(SMRegionContext):
    """The SMRegionContext a CameraSM hands to its region proposers.

    A snapshot of the module's step, taken after its goal generator ran.
    """

    def __init__(
        self,
        sensor_module_id: str,
        goals: Sequence[Goal],
        percept: Message | None,
        sensor_state: SensorState,
    ) -> None:
        """Snapshot a step.

        Args:
            sensor_module_id: The module's id.
            goals: The goals the module proposed this step.
            percept: The module's percept this step, before its percept filter.
            sensor_state: The sensor's proprioceptive state this step.
        """
        self._sensor_module_id = sensor_module_id
        self._goals = tuple(goals)
        self._percept = percept
        self._sensor_state = sensor_state

    @property
    def sensor_module_id(self) -> str:
        """The module's id."""
        return self._sensor_module_id

    @property
    def goals(self) -> Sequence[Goal]:
        """The goals the module proposed this step."""
        return self._goals

    @property
    def percept(self) -> Message | None:
        """The module's percept this step, before its percept filter."""
        return self._percept

    @property
    def sensor_state(self) -> SensorState:
        """The sensor's proprioceptive state this step."""
        return self._sensor_state
