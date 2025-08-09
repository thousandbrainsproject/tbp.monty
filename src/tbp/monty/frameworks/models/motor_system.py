# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import contextlib
from typing import Literal, Optional

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.models.states import GoalState


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self,
        policy: MotorPolicy,
        state: MotorSystemState | None = None,
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The motor policy to use.
            state: The initial state of the motor system.
                Defaults to None.
        """
        self._policy = policy
        self._state = state
        self.reset()

    @property
    def last_action(self) -> Action:
        """Returns the last action taken by the motor system."""
        return self._last_action

    def reset(self) -> None:
        """Reset the motor system."""
        self._driving_goal_state = None
        self._experiment_mode = None
        self._last_action = None

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self.reset()
        self._policy.pre_episode()

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        """Sets the driving goal state.

        Args:
            goal_state: The goal state to drive the motor system.
        """
        self._driving_goal_state = goal_state
        with contextlib.suppress(AttributeError):
            self._policy.set_driving_goal_state(goal_state)

    def set_experiment_mode(self, mode: Literal["train", "eval"]) -> None:
        """Sets the experiment mode.

        Args:
            mode: The experiment mode.
        """
        self._experiment_mode = mode
        self._policy.set_experiment_mode(mode)

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            The action to take.
        """
        self._last_action = self._policy(self._state)
        return self._last_action
