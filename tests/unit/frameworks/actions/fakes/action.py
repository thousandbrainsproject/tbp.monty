# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.actions.actuator import Actuator


class FakeAction(Action):
    """A fake action that does nothing.

    Used for testing generic functionality that uses and interfaces with Actions
    without considering specific action implementation details.
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id=agent_id)

    def act(self, _: Actuator) -> None:
        pass

    def sample(self, agent_id: str, _: ActionSampler) -> Action:
        pass
