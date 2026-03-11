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

import numpy as np
import quaternion as qt
from mujoco import Renderer, mjtJoint
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    TurnLeft,
    TurnRight,
)
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
    id: AgentID

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
        self.agent_joint = self.agent_body.add_freejoint()
        self.sensor_body = self.agent_body.add_body(
            name=f"{self.id}.sensor",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
        )
        self.pitch_joint = self.sensor_body.add_joint(
            type=mjtJoint.mjJNT_HINGE, axis=(1, 0, 0)
        )

        for sensor_id, sensor_cfg in self._sensor_configs.items():
            self.sensor_body.add_camera(
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
        # z_joint_id = self.sim.model.joint(self.z_slide_joint.id).id
        # qpos_addr = self.sim.model.jnt_qposadr[z_joint_id]
        # z_pos = self.sim.data.qpos[pos_addr]
        body_pos = self.sim.data.body(self.id).xpos
        body_quat = self.sim.data.body(self.id).xquat
        return AgentState(
            position=agent_body.pos,
            rotation=qt.quaternion(*agent_body.quat),
            sensors=sensor_states,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


class PosableAgent(NoopAgent):
    """An agent that can be moved around the scene."""

    def actuate_move_forward(self, action: MoveForward):
        body_pos = self.sim.data.body(self.id).xpos
        body_quat = self.sim.data.body(self.id).xquat
        xyzw = [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
        rotation = Rotation.from_quat(xyzw)
        rotation_matrix = rotation.as_matrix()
        forward_vector = rotation_matrix[:, 2]
        forward_vector = forward_vector / np.linalg.norm(
            forward_vector
        )  # necessary? lerarn moar math
        forward_vector = forward_vector * action.distance

        qpos_addr = self.sim.model.jnt_qposadr[self.agent_joint.id]
        cur_xyz = self.sim.data.qpos[qpos_addr : qpos_addr + 3]

        new_xyz = cur_xyz - forward_vector
        self.sim.data.qpos[qpos_addr : qpos_addr + 3] = new_xyz

    def actuate_turn_right(self, action: TurnRight):
        pass
        # radians = -np.deg2rad(action.rotation_degrees)
        # joint = self.sim.model.joint(self.yaw_joint.id)
        # qpos_addr = self.sim.model.jnt_qposadr[joint.id]
        # self.sim.data.qpos[qpos_addr] += radians

    def actuate_turn_left(self, action: TurnLeft):
        pass
        # radians = np.deg2rad(action.rotation_degrees)
        # joint = self.sim.model.joint(self.yaw_joint.id)
        # qpos_addr = self.sim.model.jnt_qposadr[joint.id]
        # self.sim.data.qpos[qpos_addr] += radians

    def actuate_look_up(self, action: LookUp):
        delta_phi = np.deg2rad(action.rotation_degrees)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] += delta_phi

    def actuate_look_down(self, action: LookDown):
        delta_phi = -np.deg2rad(action.rotation_degrees)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] += delta_phi
