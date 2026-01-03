# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState


class Simulator(Protocol):
    """A Protocol defining a simulator for use in simulated environments.

    A Simulator is responsible for a simulated environment that contains objects to
    interact with, agents to do the interacting, and for collecting observations and
    proprioceptive state to send to Monty.
    """

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Execute the given actions in the environment.

        Args:
            actions: The actions to execute.

        Returns:
            The observations from the simulator and proprioceptive state.

        Note:
            If the actions are an empty sequence, the current observations are returned.
        """
        ...

    def close(self) -> None:
        """Close any resources used by the simulator."""
        ...
