# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypedDict

import quaternion as qt
from mujoco import Renderer

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import QuaternionWXYZ, VectorXYZ

if TYPE_CHECKING:
    from tbp.monty.simulators.mujoco import MuJoCoSimulator

# TODO: Create a base Agent class?

# TODO: Move elsewhere
Size = tuple[int, int]


class Agent:
    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_ids: Sequence[SensorID],
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        positions: Sequence[VectorXYZ] = ((0.0, 0.0, 0.0),),
        rotations: Sequence[QuaternionWXYZ] = ((1.0, 0.0, 0.0, 0.0),),
        resolutions: Sequence[Size] = ((64, 64),),
        zooms: Sequence[float] = (1.0,),  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        self.id = agent_id
        self.sim = simulator

        self.position = position
        self.rotation = rotation
        self.sensor_positions = positions
        self.sensor_rotations = rotations
        self.sensor_resolutions = resolutions

        self._sensor_ids = sensor_ids

        # Create agent and sensors in MuJoCo
        self.agent_body = self.sim.spec.worldbody.add_body(
            name=agent_id,
            pos=position,
            quat=rotation,
        )
        # self.agent_body.add_joint(type=mjtJoint.mjJNT_FREE)
        for idx, sensor_id in enumerate(self._sensor_ids):
            self.agent_body.add_camera(
                name=sensor_id,
                pos=positions[idx],
                quat=rotations[idx],
                resolution=resolutions[idx],
            )

    @property
    def observations(self) -> AgentObservations:
        obs = AgentObservations()
        for idx, sensor_id in enumerate(self._sensor_ids):
            size = self.sensor_resolutions[idx]
            with Renderer(self.sim.model, width=size[0], height=size[1]) as renderer:
                renderer.update_scene(self.sim.data, camera=sensor_id)
                rbga_data = renderer.render()

                renderer.enable_depth_rendering()
                renderer.update_scene(self.sim.data, camera=sensor_id)
                depth_data = renderer.render()
                renderer.disable_depth_rendering()

                obs[sensor_id] = SensorObservation(
                    depth=depth_data,
                    rgba=rbga_data,
                )

        return obs

    @property
    def state(self) -> AgentState:
        sensor_states = {}
        for sensor_id in self._sensor_ids:
            sensor_states[sensor_id] = SensorState(
                position=(0, 0, 0),
                rotation=qt.quaternion(1, 0, 0, 0),
            )
        return AgentState(
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
            sensors=sensor_states,
        )


class AgentConfig(TypedDict):
    agent_type: type[Agent]
    agent_args: dict
