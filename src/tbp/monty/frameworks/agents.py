# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, NewType, Protocol

from tbp.monty.frameworks.sensors import Resolution2D

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import AgentObservations
    from tbp.monty.frameworks.models.motor_system_state import AgentState

__all__ = ["Agent", "AgentID"]

AgentID = NewType("AgentID", str)


class Agent(Protocol):
    """Protocol for an agent that interacts with an environment."""

    id: AgentID

    @property
    def max_sensor_resolution(self) -> Resolution2D:
        """Returns the maximum width and heights of the sensors.

        Note: the maximum width and maximum height may come from separate sensors.
        """

    @property
    def observations(self) -> AgentObservations:
        """Returns the current observations of the sensors coupled to this agent."""

    @property
    def state(self) -> AgentState:
        """Returns the current proprioceptive state of the agent."""

    def reset(self) -> None:
        """Resets the agent to its initial state."""
