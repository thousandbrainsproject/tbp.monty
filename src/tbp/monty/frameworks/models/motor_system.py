# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from typing import Any, Dict, Literal, TypedDict

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import MotorPolicy


class SensorState(TypedDict):
    """The proprioceptive state of a sensor."""

    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The sensor's position relative to the agent."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The sensor's rotation relative to the agent."""


class AgentState(TypedDict):
    """The proprioceptive state of an agent."""

    sensors: Dict[str, SensorState]
    """The proprioceptive state of the agent's sensors."""
    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The agent's position relative to some global reference frame."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The agent's rotation relative to some global reference frame."""


MotorSystemState = Dict[str, AgentState]
"""The proprioceptive state of the motor system."""


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(self, policy: MotorPolicy) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy (MotorPolicy): The motor policy to use.
        """
        self._policy = policy

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
            mode (Literal["train", "eval"]): The experiment mode.
        """
        self._policy.set_experiment_mode(mode)

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            (Action): The action to take.
        """
        action = self._policy()
        return action
