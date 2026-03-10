# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

import mujoco
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

# The default field of view value for zoom 1.0
# Note: this value is the half-FOV rather than the full FOV
DEFAULT_CAMERA_FOVY: float = 45.0


class SensorConfig(TypedDict):
    position: VectorXYZ
    rotation: QuaternionWXYZ
    resolution: Size
    zoom: float


class AgentConfig(TypedDict):
    agent_type: type[Agent]
    agent_args: dict


class Agent(Protocol):
    @property
    def observations(self) -> AgentObservations: ...

    @property
    def state(self) -> AgentState: ...


class NoopAgent(Agent):
    """A simple multi-sensor agent that doesn't respond to actions."""

    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_configs: dict[SensorID, SensorConfig],
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
    ):
        self.id = agent_id
        self.sim = simulator

        self.position = position
        self.rotation = rotation
        self._sensor_configs = sensor_configs

        # Create agent and sensors in MuJoCo
        self.agent_body = self.sim.spec.worldbody.add_body(
            name=agent_id,
            pos=position,
            quat=rotation,
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
        )
        # self.agent_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        freejoint = self.agent_body.add_freejoint()
        for sensor_id, sensor_cfg in self._sensor_configs.items():
            self.agent_body.add_camera(
                name=f"{self.id}.{sensor_id}",
                pos=sensor_cfg["position"],
                quat=sensor_cfg["rotation"],
                resolution=sensor_cfg["resolution"],
                fovy=DEFAULT_CAMERA_FOVY / sensor_cfg["zoom"],
            )

    @property
    def observations(self) -> AgentObservations:
        obs = AgentObservations()
        for sensor_id, sensor_cfg in self._sensor_configs.items():
            size = sensor_cfg["resolution"]
            with Renderer(self.sim.model, width=size[0], height=size[1]) as renderer:
                renderer.update_scene(self.sim.data, camera=f"{self.id}.{sensor_id}")
                rbga_data = renderer.render()

                renderer.enable_depth_rendering()
                renderer.update_scene(self.sim.data, camera=f"{self.id}.{sensor_id}")
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
        for sensor_id in self._sensor_configs:
            sensor = self.sim.model.camera(f"{self.id}.{sensor_id}")
            sensor_states[sensor_id] = SensorState(
                position=sensor.pos,
                rotation=qt.quaternion(*sensor.quat),
            )
        agent_body = self.sim.model.body(self.id)
        return AgentState(
            position=agent_body.pos,
            rotation=qt.quaternion(*agent_body.quat),
            sensors=sensor_states,
        )
