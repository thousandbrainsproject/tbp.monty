# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Literal

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.models.states import GoalState, State


class MotorSystem:
    """The basic motor system implementation.

    Attributes:
        _policy: The current motor policy to use.
        _state: The motor system state.
        _driving_goal_state: The goal state to drive the motor system.
        _experiment_mode: The experiment mode.
        _last_action: The last action taken by the motor system.
        _processed_observations: The processed observations.

    """

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
        self.reset()
        if state:
            self._state = state

    @property
    def agent_id(self) -> str:
        """Returns the agent ID of the motor system."""
        return self._policy.agent_id

    @property
    def last_action(self) -> Action | None:
        """Returns the last action taken by the motor system."""
        return self._last_action

    @property
    def policy(self) -> MotorPolicy:
        """Returns the motor policy."""
        return self._policy

    @property
    def state(self) -> MotorSystemState:
        """Returns the state of the motor system."""
        return self._state

    @state.setter
    def state(self, state: MotorSystemState | None) -> None:
        """Sets the state of the motor system."""
        self._state = state

    def driving_goal_state(self) -> GoalState | None:
        """Returns the driving goal state."""
        return self._driving_goal_state

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        """Sets the driving goal state.

        Args:
            goal_state: The goal state to drive the motor system.
        """
        self._driving_goal_state = goal_state

    def experiment_mode(self) -> Literal["train", "eval"] | None:
        """Returns the experiment mode."""
        return self._experiment_mode

    def set_experiment_mode(self, mode: Literal["train", "eval"] | None) -> None:
        """Sets the experiment mode."""
        self._experiment_mode = mode

    def processed_observations(self) -> State | None:
        """Returns the processed observations."""
        return self._processed_observations

    def set_processed_observations(self, processed_observations: State | None) -> None:
        """Sets the processed observations."""
        self._processed_observations = processed_observations

    def reset(self) -> None:
        """Reset the motor system."""
        self._state = None
        self._driving_goal_state = None
        self._experiment_mode = None
        self._processed_observations = None
        self._last_action = None

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self.reset()
        self._policy.pre_episode()

    def step(self) -> None:
        """Select a policy, etc.

        This must be called before `__call__()` is used.

        The important thing here is to determine whether the driving goal state
        should be attempted with the data loader's execute_jump_attempt() method.
        If so, we need to set self._policy to the appropriate `InformedPolicy` object
        and then set its `driving_goal_state` attribute. This is what the data loader
        will look for when deciding to use `execute_jump_attempt()`.

        If we don't want to attempt the driving goal state with a jump, we need to
        set self._policy to some other policy but maybe set an attribute like
        `driving_goal_state` but with a different name (or else not have that policy
        inherit from `InformedPolicy`).

        If there is no driving goal state, pick some other policy.
        """
        self._policy = self._select_policy()
        self._policy.set_experiment_mode(self._experiment_mode)
        if hasattr(self._policy, "set_driving_goal_state"):
            self._policy.set_driving_goal_state(self._driving_goal_state)
        self._policy.processed_observations = self._processed_observations

    def _select_policy(self) -> MotorPolicy:
        """Selects a policy for the motor system.

        Returns:
            The policy to use.
        """
        return self._policy

    def _post_call(self) -> None:
        """Post call hook."""
        # ?Need to keep this in sync with the policy's driving goal state since
        # derive_habitat_goal_state() consumes the goal state.?
        self._driving_goal_state = getattr(self._policy, "driving_goal_state", None)

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            The action to take.
        """
        # TODO: ?Mark a goal state being attempted as the one being attempted so
        # it can be checked by a GSG.?
        action = self._policy(self._state)
        self._last_action = action

        self._post_call()

        return action
