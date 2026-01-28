# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Callable

import quaternion as qt
from numpy import cos, pi, sin, sqrt
from numpy.random import RandomState

from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    QuaternionWXYZ,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
    VectorXYZ,
)
from tbp.monty.frameworks.agents import AgentID

__all__ = [
    "ActionSampler",
    "ConstantSampler",
    "UniformlyDistributedSampler",
]


class ActionSampler:
    """An Action factory that samples from a set of available action types."""

    def __init__(
        self,
        actions: list[type[Action]] | None = None,
    ):
        self._actions: list[type[Action]] = actions if actions is not None else []
        self._action_names: list[str] = [
            action.action_name() for action in self._actions
        ]
        self._method_names: list[str] = [
            f"sample_{action_name}" for action_name in self._action_names
        ]

    def sample(self, agent_id: AgentID, rng: RandomState) -> Action:
        """Sample a random action from the available action types.

        Returns:
            Action: A random action from the available action types.
        """
        random_create_method_name: str = rng.choice(self._method_names)
        random_create_method: Callable[[str, RandomState], Action] = getattr(
            self, random_create_method_name
        )
        return random_create_method(agent_id, rng)


class ConstantSampler(ActionSampler):
    """An Action factory using constant, prespecified action parameters.

    This Action factory samples actions with constant parameters.
    The values of action parameters used are set at initialization time and
    remain the same for all actions created by this factory. For example,
    if you specify `rotation_degrees=5.0`, all actions created by this factory
    that take a `rotation_degrees` parameter will have it set to `5.0`.

    When sampling an Action, only applicable parameters are used. For example,
    when sampling a `MoveForward` action, only the ConstantSampler's
    `translation_distance` parameter is used to determine the action's `distance`
    parameter.
    """

    def __init__(
        self,
        absolute_degrees: float = 0.0,
        actions: list[type[Action]] | None = None,
        direction: VectorXYZ | None = None,
        location: VectorXYZ | None = None,
        rotation_degrees: float = 5.0,
        rotation_quat: QuaternionWXYZ | None = None,
        translation_distance: float = 0.004,
    ) -> None:
        super().__init__(actions=actions)
        self.absolute_degrees = absolute_degrees
        self.direction = direction if direction is not None else (0.0, 0.0, 0.0)
        self.location = location if location is not None else (0.0, 0.0, 0.0)
        self.rotation_degrees = rotation_degrees
        self.rotation_quat = rotation_quat if rotation_quat is not None else qt.one
        self.translation_distance = translation_distance

    def sample_look_down(self, agent_id: AgentID, rng: RandomState) -> LookDown:  # noqa: ARG002
        return LookDown(agent_id=agent_id, rotation_degrees=self.rotation_degrees)

    def sample_look_up(self, agent_id: AgentID, rng: RandomState) -> LookUp:  # noqa: ARG002
        return LookUp(agent_id=agent_id, rotation_degrees=self.rotation_degrees)

    def sample_move_forward(self, agent_id: AgentID, rng: RandomState) -> MoveForward:  # noqa: ARG002
        return MoveForward(agent_id=agent_id, distance=self.translation_distance)

    def sample_move_tangentially(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> MoveTangentially:
        return MoveTangentially(
            agent_id=agent_id,
            distance=self.translation_distance,
            direction=self.direction,
        )

    def sample_orient_horizontal(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> OrientHorizontal:
        return OrientHorizontal(
            agent_id=agent_id,
            rotation_degrees=self.rotation_degrees,
            left_distance=self.translation_distance,
            forward_distance=self.translation_distance,
        )

    def sample_orient_vertical(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> OrientVertical:
        return OrientVertical(
            agent_id=agent_id,
            rotation_degrees=self.rotation_degrees,
            down_distance=self.translation_distance,
            forward_distance=self.translation_distance,
        )

    def sample_set_agent_pitch(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> SetAgentPitch:
        return SetAgentPitch(agent_id=agent_id, pitch_degrees=self.absolute_degrees)

    def sample_set_agent_pose(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> SetAgentPose:
        return SetAgentPose(
            agent_id=agent_id, location=self.location, rotation_quat=self.rotation_quat
        )

    def sample_set_sensor_pitch(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> SetSensorPitch:
        return SetSensorPitch(agent_id=agent_id, pitch_degrees=self.absolute_degrees)

    def sample_set_sensor_pose(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> SetSensorPose:
        return SetSensorPose(
            agent_id=agent_id, location=self.location, rotation_quat=self.rotation_quat
        )

    def sample_set_sensor_rotation(
        self,
        agent_id: AgentID,
        rng: RandomState,  # noqa: ARG002
    ) -> SetSensorRotation:
        return SetSensorRotation(agent_id=agent_id, rotation_quat=self.rotation_quat)

    def sample_set_yaw(self, agent_id: AgentID, rng: RandomState) -> SetYaw:  # noqa: ARG002
        return SetYaw(agent_id=agent_id, rotation_degrees=self.absolute_degrees)

    def sample_turn_left(self, agent_id: AgentID, rng: RandomState) -> TurnLeft:  # noqa: ARG002
        return TurnLeft(agent_id=agent_id, rotation_degrees=self.rotation_degrees)

    def sample_turn_right(self, agent_id: AgentID, rng: RandomState) -> TurnRight:  # noqa: ARG002
        return TurnRight(agent_id=agent_id, rotation_degrees=self.rotation_degrees)


class UniformlyDistributedSampler(ActionSampler):
    """An Action factory using uniformly distributed action creation parameters.

    This Action factory samples actions with parameters that are uniformly
    distributed within a given range. The range of values for each parameter is set
    at initialization time and remains the same for all actions created by this factory.

    When sampling an Action, only applicable parameters are used. For example,
    when sampling a `MoveForward` action, only the UniformlyDistributedSampler's
    `translation_high` and `translation_low` parameters are used to determine the
    action's `distance` parameter.
    """

    def __init__(
        self,
        actions: list[type[Action]] | None = None,
        max_absolute_degrees: float = 360.0,
        min_absolute_degrees: float = 0.0,
        max_rotation_degrees: float = 20.0,
        min_rotation_degrees: float = 0.0,
        max_translation: float = 0.05,
        min_translation: float = 0.05,
    ):
        super().__init__(actions=actions)
        self.max_absolute_degrees = max_absolute_degrees
        self.min_absolute_degrees = min_absolute_degrees
        self.max_rotation_degrees = max_rotation_degrees
        self.min_rotation_degrees = min_rotation_degrees
        self.max_translation = max_translation
        self.min_translation = min_translation

    def _random_quaternion_wxyz(self, rng: RandomState) -> QuaternionWXYZ:
        u, v, w = rng.random(3)
        return (
            sqrt(1 - u) * sin(2 * pi * v),
            sqrt(1 - u) * cos(2 * pi * v),
            sqrt(u) * sin(2 * pi * w),
            sqrt(u) * cos(2 * pi * w),
        )

    def _random_vector_xyz(self, rng: RandomState) -> VectorXYZ:
        return (rng.random(), rng.random(), rng.random())

    def sample_look_down(self, agent_id: AgentID, rng: RandomState) -> LookDown:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        return LookDown(agent_id=agent_id, rotation_degrees=rotation_degrees)

    def sample_look_up(self, agent_id: AgentID, rng: RandomState) -> LookUp:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        return LookUp(agent_id=agent_id, rotation_degrees=rotation_degrees)

    def sample_move_forward(self, agent_id: AgentID, rng: RandomState) -> MoveForward:
        distance = rng.uniform(low=self.min_translation, high=self.max_translation)
        return MoveForward(agent_id=agent_id, distance=distance)

    def sample_move_tangentially(
        self, agent_id: AgentID, rng: RandomState
    ) -> MoveTangentially:
        distance = rng.uniform(low=self.min_translation, high=self.max_translation)
        direction = self._random_vector_xyz(rng)
        return MoveTangentially(
            agent_id=agent_id,
            distance=distance,
            direction=direction,
        )

    def sample_orient_horizontal(
        self, agent_id: AgentID, rng: RandomState
    ) -> OrientHorizontal:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        left_distance = rng.uniform(low=self.min_translation, high=self.max_translation)
        forward_distance = rng.uniform(
            low=self.min_translation, high=self.max_translation
        )
        return OrientHorizontal(
            agent_id=agent_id,
            rotation_degrees=rotation_degrees,
            left_distance=left_distance,
            forward_distance=forward_distance,
        )

    def sample_orient_vertical(
        self, agent_id: AgentID, rng: RandomState
    ) -> OrientVertical:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        down_distance = rng.uniform(low=self.min_translation, high=self.max_translation)
        forward_distance = rng.uniform(
            low=self.min_translation, high=self.max_translation
        )
        return OrientVertical(
            agent_id=agent_id,
            rotation_degrees=rotation_degrees,
            down_distance=down_distance,
            forward_distance=forward_distance,
        )

    def sample_set_agent_pitch(
        self, agent_id: AgentID, rng: RandomState
    ) -> SetAgentPitch:
        pitch_degrees = rng.uniform(
            low=self.min_absolute_degrees, high=self.max_absolute_degrees
        )
        return SetAgentPitch(agent_id=agent_id, pitch_degrees=pitch_degrees)

    def sample_set_agent_pose(
        self, agent_id: AgentID, rng: RandomState
    ) -> SetAgentPose:
        location = self._random_vector_xyz(rng)
        rotation_quat = self._random_quaternion_wxyz(rng)
        return SetAgentPose(
            agent_id=agent_id, location=location, rotation_quat=rotation_quat
        )

    def sample_set_sensor_pitch(
        self, agent_id: AgentID, rng: RandomState
    ) -> SetSensorPitch:
        pitch_degrees = rng.uniform(
            low=self.min_absolute_degrees, high=self.max_absolute_degrees
        )
        return SetSensorPitch(agent_id=agent_id, pitch_degrees=pitch_degrees)

    def sample_set_sensor_pose(
        self, agent_id: AgentID, rng: RandomState
    ) -> SetSensorPose:
        location = self._random_vector_xyz(rng)
        rotation_quat = self._random_quaternion_wxyz(rng)
        return SetSensorPose(
            agent_id=agent_id, location=location, rotation_quat=rotation_quat
        )

    def sample_set_sensor_rotation(
        self, agent_id: AgentID, rng: RandomState
    ) -> SetSensorRotation:
        rotation_quat = self._random_quaternion_wxyz(rng)
        return SetSensorRotation(agent_id=agent_id, rotation_quat=rotation_quat)

    def sample_set_yaw(self, agent_id: AgentID, rng: RandomState) -> SetYaw:
        rotation_degrees = rng.uniform(
            low=self.min_absolute_degrees, high=self.max_absolute_degrees
        )
        return SetYaw(agent_id=agent_id, rotation_degrees=rotation_degrees)

    def sample_turn_left(self, agent_id: AgentID, rng: RandomState) -> TurnLeft:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        return TurnLeft(agent_id=agent_id, rotation_degrees=rotation_degrees)

    def sample_turn_right(self, agent_id: AgentID, rng: RandomState) -> TurnRight:
        rotation_degrees = rng.uniform(
            low=self.min_rotation_degrees, high=self.max_rotation_degrees
        )
        return TurnRight(agent_id=agent_id, rotation_degrees=rotation_degrees)
