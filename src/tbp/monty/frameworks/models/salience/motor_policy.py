# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import LookUp, TurnLeft
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy, MotorPolicyResult
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

logger = logging.getLogger(__name__)


class LookAtGoal(MotorPolicy):
    """A policy that looks at a target.

    This class assumes a system similar to a 2-DOF gimbal in which the "outer" part
    can yaw left/right about the y-axis and the "inner" part can pitch up/down about
    the x-axis. This setup is typical of our distant agent in which the agent
    performs TurnLeft and TurnRight, while the sensor mounted to it performs
    LookDown and LookUp.
    """

    def __init__(self, agent_id: AgentID, sensor_id: SensorID):
        """Initialize the look at policy.

        Args:
            agent_id: The agent ID
            sensor_id: The sensor ID
        """
        self._agent_id = agent_id
        self._sensor_id = sensor_id

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def pre_episode(self, motor_system: MotorSystem) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observations: Observations,  # noqa: ARG002
        state: MotorSystemState,
        percept: Message,  # noqa: ARG002
        goal: Goal | None,
    ) -> MotorPolicyResult:
        """Invoke motor policy to determine the next actions to take.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            state: The current state of the motor system.
            percept: The percept from (as of this writing) the first sensor
                module.
            goal: The goal to look at (in world reference frame).

        Returns:
            The motor policy result.

        Raises:
            RuntimeError: If no goal is provided.
        """
        if goal is None:
            raise RuntimeError("No goal provided")

        # Collect necessary agent and sensor pose information.
        agent_state: AgentState = state[self._agent_id]
        agent_pos_rel_world = agent_state.position
        agent_rot_rel_world = Rotation.from_quat(
            [
                agent_state.rotation.x,
                agent_state.rotation.y,
                agent_state.rotation.z,
                agent_state.rotation.w,
            ]
        )

        sensor_state: SensorState = agent_state.sensors[self._sensor_id]
        sensor_rot_rel_agent = Rotation.from_quat(
            [
                sensor_state.rotation.x,
                sensor_state.rotation.y,
                sensor_state.rotation.z,
                sensor_state.rotation.w,
            ]
        )

        # Get the target location in world and agent coordinates.
        target_rel_world = goal.location
        target_rel_agent = agent_rot_rel_world.apply(
            target_rel_world - agent_pos_rel_world,
            inverse=True,
        )

        # Compute the target's azimuth, relative to the agent. This value is used to
        # compute the yaw action to be performed by the agent.
        agent_yaw = -np.arctan2(target_rel_agent[0], -target_rel_agent[2])

        # Compute the target's elevation, relative to the agent. Then subtract the
        # sensor's current pitch to get a pitch delta effective for the sensor. This
        # value is used to compute the look up/down action which must be performed
        # by the sensor mounted to the agent.
        target_pitch_rel_agent = np.arctan2(
            target_rel_agent[1], np.hypot(target_rel_agent[0], target_rel_agent[2])
        )
        sensor_pitch_rel_agent = sensor_rot_rel_agent.as_euler("xyz")[0]
        sensor_pitch = target_pitch_rel_agent - sensor_pitch_rel_agent

        # Create actions to return to the the motor system.
        actions = [
            TurnLeft(agent_id=self._agent_id, rotation_degrees=np.degrees(agent_yaw)),
            LookUp(agent_id=self._agent_id, rotation_degrees=np.degrees(sensor_pitch)),
        ]

        return MotorPolicyResult(actions)
