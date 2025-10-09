# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Literal, Optional

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self, policy: MotorPolicy, state: Optional[MotorSystemState] = None
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The motor policy to use.
            state: The initial state of the motor system.
                Defaults to None.
        """
        self._policy = policy
        self._state = state

    @property
    def last_action(self) -> Action:
        """Returns the last action taken by the motor system."""
        return self._policy.last_action

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self._policy.pre_episode()

    def set_experiment_mode(self, mode: Literal["train", "eval"]) -> None:
        """Sets the experiment mode.

        Args:
            mode: The experiment mode.
        """
        self._policy.set_experiment_mode(mode)

    def __call__(self) -> list[Action]:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            The action to take.
        """
        actions = self._policy(self._state)
        return actions
