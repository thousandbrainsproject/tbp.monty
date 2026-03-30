# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, NewType, Protocol, TypedDict

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import AgentObservations
    from tbp.monty.frameworks.models.motor_system_state import AgentState

__all__ = ["Agent", "AgentConfig", "AgentID"]

AgentID = NewType("AgentID", str)


class Agent(Protocol):
    """Protocol for an agent that interacts with an environment."""

    id: AgentID

    @property
    def observations(self) -> AgentObservations:
        """Returns the current observations of the agent."""

    @property
    def state(self) -> AgentState:
        """Returns the current proprioceptive state of the agent."""

    def reset(self) -> None:
        """Resets the agent to its initial state."""


class AgentConfig(TypedDict):
    """The configuration for an agent, mapping to our configs in Hydra."""

    agent_type: type[Agent]
    agent_args: dict  # TODO: be more specific
