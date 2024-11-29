# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from habitat_sim import Agent

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
from tbp.monty.frameworks.actions.actuator import Actuator

__all__ = [
    "HabitatActuator",
    "HabitatActuatorRequirements",
    "HabitatParameterizer",
]


class HabitatActuatorRequirements(ABC):
    """HabitatActuator requires these to be available when mixed in."""

    @abstractmethod
    def get_agent(self, agent_id: str) -> Agent:
        pass


class HabitatActuator(Actuator, HabitatActuatorRequirements):
    """Habitat implementation of Actuator.

    HabitatActuator is responsible for executing actions in the Habitat simulation.

    It is a separate class to encapsulate the actuation logic in one place. This
    class is expected to be mixed into HabitatSim and expects
    HabitatActuatorRequirements to be met.
    """

    def action_name(self, action: Action) -> str:
        """Returns Monty's Habitat action naming convention.

        The action name is prefixed by the agent ID.
        """
        return f"{action.agent_id}.{action.name}"

    def actuate(self, action: Action, parameterizer: Type[HabitatParameterizer]):
        """Transition from the Monty to the Habitat sim domain and execute the action.

        Args:
            action: Monty action to execute by the agent specified in the action.
            parameterizer: Parameterizer to use to set action parameters within Habitat
                sim prior to executing the action.

        Raises:
            ValueError: If the action name is invalid
        """
        agent = self.get_agent(action.agent_id)
        action_name = self.action_name(action)
        action_space = agent.agent_config.action_space
        if action_name not in action_space:
            raise ValueError(f"Invalid action name: {action_name}")

        # actuation is Habitat's name for action parameters when action is executed
        action_params = action_space[action_name].actuation
        # overwrite Habitat action parameters with values from Monty action
        parameterizer.parameterize(action_params, action)

        agent.act(action_name)

    def actuate_look_down(self, action: LookDown) -> None:
        self.actuate(action, LookDownParameterizer)

    def actuate_look_up(self, action: LookUp) -> None:
        self.actuate(action, LookUpParameterizer)

    def actuate_move_forward(self, action: MoveForward) -> None:
        self.actuate(action, MoveForwardParameterizer)

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        self.actuate(action, MoveTangentiallyParameterizer)

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        self.actuate(action, OrientHoriztonalParameterizer)

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        self.actuate(action, OrientVerticalParameterizer)

    def actuate_set_agent_pitch(self, action: SetAgentPitch) -> None:
        self.actuate(action, SetAgentPitchParameterizer)

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        self.actuate(action, SetAgentPoseParameterizer)

    def actuate_set_sensor_pitch(self, action: SetSensorPitch) -> None:
        self.actuate(action, SetSensorPitchParameterizer)

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        self.actuate(action, SetSensorPoseParameterizer)

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        self.actuate(action, SetSensorRotationParameterizer)

    def actuate_set_yaw(self, action: SetYaw) -> None:
        self.actuate(action, SetYawParameterizer)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        self.actuate(action, TurnLeftParameterizer)

    def actuate_turn_right(self, action: TurnRight) -> None:
        self.actuate(action, TurnRightParameterizer)


class HabitatParameterizer(ABC):
    @staticmethod
    @abstractmethod
    def parameterize(params: Any, action: Action) -> None:
        """Copies relevant parameters from action to params.

        Habitat does not expose an API for passing parameters to actions.
        This is a work around for this limitation by artisanally setting
        specific action parameters directly in Habitat sim.
        """
        pass


class LookDownParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: LookDown) -> None:
        params.amount = action.rotation_degrees
        params.constraint = action.constraint_degrees


class LookUpParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: LookUp) -> None:
        params.amount = action.rotation_degrees
        params.constraint = action.constraint_degrees


class MoveForwardParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: MoveForward) -> None:
        params.amount = action.distance


class MoveTangentiallyParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: MoveTangentially) -> None:
        params.amount = action.distance
        params.constraint = action.direction


class OrientHoriztonalParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: OrientHorizontal) -> None:
        params.amount = action.rotation_degrees
        params.constraint = [action.left_distance, action.forward_distance]


class OrientVerticalParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: OrientVertical) -> None:
        params.amount = action.rotation_degrees
        params.constraint = [action.down_distance, action.forward_distance]


class SetAgentPitchParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetAgentPitch) -> None:
        params.amount = action.pitch_degrees


class SetAgentPoseParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetAgentPose) -> None:
        params.amount = [action.location, action.rotation_quat]


class SetSensorPitchParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetSensorPitch) -> None:
        params.amount = action.pitch_degrees


class SetSensorPoseParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetSensorPose) -> None:
        params.amount = [action.location, action.rotation_quat]


class SetSensorRotationParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetSensorRotation) -> None:
        params.amount = [action.rotation_quat]


class SetYawParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: SetYaw) -> None:
        params.amount = action.rotation_degrees


class TurnLeftParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: TurnLeft) -> None:
        params.amount = action.rotation_degrees


class TurnRightParameterizer(HabitatParameterizer):
    @staticmethod
    def parameterize(params: Any, action: TurnRight) -> None:
        params.amount = action.rotation_degrees
