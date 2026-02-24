# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

__all__ = ["MotorSystem"]


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self, policy: MotorPolicy, state: MotorSystemState | None = None
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The motor policy to use.
            state: The initial state of the motor system.
                Defaults to None.
        """
        self._policy = policy
        self._state = state
        # For each step, we store the actions produced by the policy and the current
        # motor system state as a (actions, state) tuple.
        self._action_sequence: list[tuple[list[Action], dict[AgentID, Any] | None]] = []

    @property
    def action_sequence(self) -> list[tuple[list[Action], dict[AgentID, Any] | None]]:
        return self._action_sequence

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self._policy.pre_episode()
        self._action_sequence = []

    def __call__(self, ctx: RuntimeContext) -> list[Action]:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Args:
            ctx: The runtime context.

        Returns:
            The action to take.
        """
        policy_result = self._policy(ctx, self._state)
        state_copy = self._state.convert_motor_state() if self._state else None
        self._action_sequence.append((policy_result.actions, state_copy))
        return policy_result.actions
