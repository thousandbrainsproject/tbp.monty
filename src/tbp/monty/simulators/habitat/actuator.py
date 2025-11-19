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

from habitat_sim import ActuationSpec, Agent
from typing_extensions import Protocol

from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID

__all__ = [
    "HabitatActuator",
    "HabitatActuatorRequirements",
]


class HabitatActuatorRequirements(Protocol):
    """HabitatActuator requires these to be available when mixed in."""

    def get_agent(self, agent_id: AgentID) -> Agent | None: ...


class HabitatActuator(HabitatActuatorRequirements):
    """Habitat implementation of an Actuator.

    HabitatActuator is responsible for executing actions in the Habitat simulation.

    It is a separate class to encapsulate the actuation logic in one place. This
    class is expected to be mixed into HabitatSim and expects
    HabitatActuatorRequirements to be met.

    Note:
        Habitat does not expose an API for passing parameters to actions.
        So each actuate method works around this limitation by artisanally setting
        specific action parameters directly in Habitat sim.
    """

    def action_name(self, action: Action) -> str:
        """Returns Monty's Habitat action naming convention.

        The action name is prefixed by the agent ID.
        """
        return f"{action.agent_id}.{action.name}"

    def to_habitat(self, action: Action) -> tuple[Agent, ActuationSpec, str]:
        """Transition from the Monty to the Habitat sim domain.

        Args:
            action: Monty action to execute by the agent specified in the action.

        Returns:
            The Habitat agent to execute the action, the Habitat action parameters to
            set prior to executing the action, and the Habitat action name to execute.

        Raises:
            InvalidActionName: If the action name is invalid.
            NoActionParameters: If the action has no parameters.
        """
        agent = self.get_agent(action.agent_id)
        action_name = self.action_name(action)
        action_space = agent.agent_config.action_space
        if action_name not in action_space:
            raise InvalidActionName(action_name)

        # actuation is Habitat's name for action parameters when action is executed
        action_params = action_space[action_name].actuation
        if action_params is None:
            raise NoActionParameters(action_name)

        return agent, action_params, action_name

    def actuate_look_down(self, action: LookDown) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        action_params.constraint = action.constraint_degrees
        agent.act(action_name)

    def actuate_look_up(self, action: LookUp) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        action_params.constraint = action.constraint_degrees
        agent.act(action_name)

    def actuate_move_forward(self, action: MoveForward) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.distance
        agent.act(action_name)

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.distance
        action_params.constraint = action.direction
        agent.act(action_name)

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        action_params.constraint = [action.left_distance, action.forward_distance]
        agent.act(action_name)

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        action_params.constraint = [action.down_distance, action.forward_distance]
        agent.act(action_name)

    def actuate_set_agent_pitch(self, action: SetAgentPitch) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.pitch_degrees
        agent.act(action_name)

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = [action.location, action.rotation_quat]
        agent.act(action_name)

    def actuate_set_sensor_pitch(self, action: SetSensorPitch) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.pitch_degrees
        agent.act(action_name)

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = [action.location, action.rotation_quat]
        agent.act(action_name)

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = [action.rotation_quat]
        agent.act(action_name)

    def actuate_set_yaw(self, action: SetYaw) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        agent.act(action_name)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        agent.act(action_name)

    def actuate_turn_right(self, action: TurnRight) -> None:
        agent, action_params, action_name = self.to_habitat(action)
        action_params.amount = action.rotation_degrees
        agent.act(action_name)


class InvalidActionName(Exception):
    """Raised when an action name is invalid."""

    def __init__(self, action_name: str):
        super().__init__(f"Invalid action name: {action_name}")


class NoActionParameters(Exception):
    """Raised when an action has no parameters."""

    def __init__(self, action_name: str):
        super().__init__(f"No action parameters for action: {action_name}")
