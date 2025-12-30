# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing_extensions import Protocol

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID

__all__ = ["FakeAction", "FakeActionActionSampler", "FakeActionActuator"]


class FakeActionActionSampler(Protocol):
    def sample_fake_action(self, agent_id: AgentID) -> FakeAction: ...


class FakeActionActuator(Protocol):
    def actuate_fake_action(self, action: FakeAction) -> None: ...


class FakeAction(Action):
    """A fake action that does nothing.

    Used for testing generic functionality that uses and interfaces with Actions
    without considering specific action implementation details.
    """

    @classmethod
    def sample(cls, agent_id: AgentID, sampler: FakeActionActionSampler) -> FakeAction:
        return sampler.sample_fake_action(agent_id)

    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id=agent_id)

    def act(self, _: FakeActionActuator) -> None:
        pass
