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
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np
import quaternion as qt
from mujoco import MjsBody, Renderer, mjtJoint
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
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
from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
    rotation_from_quat,
)
from tbp.monty.math import QuaternionWXYZ, VectorXYZ

if TYPE_CHECKING:
    from tbp.monty.simulators.mujoco import MuJoCoSimulator

logger = logging.getLogger(__name__)

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

    def reset(self) -> None: ...


class AgentBase(Agent):
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

        self._initial_position = position
        self._initial_rotation = rotation
        self._sensor_configs = sensor_configs

        # Create agent and sensors in MuJoCo
        agent_body: MjsBody = self.sim.spec.worldbody.add_body(
            name=agent_id,
            pos=position,
            quat=rotation,
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
        )
        self.agent_joint = agent_body.add_freejoint()
        sensor_body: MjsBody = agent_body.add_body(
            name=f"{self.id}.sensor",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
        )
        self.pitch_joint = sensor_body.add_joint(
            type=mjtJoint.mjJNT_HINGE, axis=(1, 0, 0)
        )
        for sensor_id, sensor_cfg in self._sensor_configs.items():
            sensor_body.add_camera(
                name=f"{self.id}.{sensor_id}",
                pos=sensor_cfg["position"],
                quat=sensor_cfg["rotation"],
                resolution=sensor_cfg["resolution"],
                fovy=DEFAULT_CAMERA_FOVY / sensor_cfg["zoom"],
            )

    @property
    def position(self) -> VectorXYZ:
        qpos_addr = self.sim.model.jnt_qposadr[self.agent_joint.id]
        return cast("VectorXYZ", tuple(self.sim.data.qpos[qpos_addr : qpos_addr + 3]))

    @position.setter
    def position(self, position: VectorXYZ) -> None:
        qpos_addr = self.sim.model.jnt_qposadr[self.agent_joint.id]
        self.sim.data.qpos[qpos_addr : qpos_addr + 3] = np.array(position)

    @property
    def rotation(self) -> QuaternionWXYZ:
        qpos_addr = self.sim.model.jnt_qposadr[self.agent_joint.id]
        return cast(
            "QuaternionWXYZ", tuple(self.sim.data.qpos[qpos_addr + 3 : qpos_addr + 7])
        )

    @rotation.setter
    def rotation(self, rotation: QuaternionWXYZ) -> None:
        qpos_addr = self.sim.model.jnt_qposadr[self.agent_joint.id]
        self.sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(rotation)

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
        """Get the state of the agent."""
        # Get agent position and rotation. Both are in world coordinates.
        agent_pos = self.sim.data.body(self.id).xpos.copy()
        agent_quat = self.sim.data.body(self.id).xquat
        agent_rotation = rotation_from_quat(agent_quat)

        # Get sensor position and rotation relative to the agent.
        # Note: The sensor body position is in world coordinates.
        pitch_body_rot = rotation_from_quat(
            self.sim.data.body(f"{self.id}.sensor").xquat
        )
        pitch_body_rot_rel_agent = agent_rotation.inv() * pitch_body_rot
        pitch_body_rot_quat = qt.quaternion(*rotation_as_quat(pitch_body_rot_rel_agent))

        sensor_states = {}
        for sensor_id, sensor_config in self._sensor_configs.items():
            sensor_pos_rel_agent = pitch_body_rot_rel_agent.apply(
                sensor_config["position"]
            )
            sensor_states[sensor_id] = SensorState(
                position=cast("VectorXYZ", tuple(sensor_pos_rel_agent)),
                rotation=pitch_body_rot_quat,
            )
        state = AgentState(
            position=agent_pos,
            rotation=qt.quaternion(*rotation_as_quat(agent_rotation)),
            sensors=sensor_states,
        )
        logger.debug(f"{state=}")
        return state

    def reset(self) -> None:
        self.position = self._initial_position
        self.rotation = self._initial_rotation

    def actuate_set_agent_pose(self, action: SetAgentPose):
        self.position = action.location
        self.rotation = qt.as_float_array(action.rotation_quat)

    def actuate_set_sensor_rotation(self, action: SetSensorRotation):
        rotation = rotation_from_quat(qt.as_float_array(action.rotation_quat))
        angles = rotation.as_euler("xyz", degrees=False)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] = angles[0]

    def _actuate_yaw(self, delta_theta: float):
        """Yaw the agent body by a specified number of degrees.

        Args:
            delta_theta: The number of degrees to yaw the agent body.
        """
        delta_theta_rot = Rotation.from_euler("xyz", (0, delta_theta, 0), degrees=True)
        rotation = rotation_from_quat(self.rotation)
        new_rotation = rotation * delta_theta_rot
        self.rotation = rotation_as_quat(new_rotation)

    def _actuate_pitch(self, delta_phi: float):
        """Pitch the sensor body by a specified number of degrees.

        Args:
            delta_phi: The number of degrees to pitch the sensor body.
        """
        delta_phi = np.deg2rad(delta_phi)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] += delta_phi

    def _move_along_local_axis(self, distance: float, axis: int):
        rotation = rotation_from_quat(self.rotation)
        rotation_matrix = rotation.as_matrix()
        axis_vector = rotation_matrix[:, axis] * distance
        new_xyz = np.array(self.position) + axis_vector
        self.position = new_xyz

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


class DistantAgent(AgentBase):
    """A multi-sensor agent for sensing an object from a distance."""

    def actuate_move_forward(self, action: MoveForward):
        self._move_along_local_axis(-action.distance, 2)

    def actuate_turn_right(self, action: TurnRight):
        self._actuate_yaw(-action.rotation_degrees)

    def actuate_turn_left(self, action: TurnLeft):
        self._actuate_yaw(action.rotation_degrees)

    def actuate_look_up(self, action: LookUp):
        self._actuate_pitch(action.rotation_degrees)

    def actuate_look_down(self, action: LookDown):
        self._actuate_pitch(-action.rotation_degrees)


class SurfaceAgent(AgentBase):
    """A multi-sensor agent for sensing an object from a distance."""

    def actuate_move_forward(self, action: MoveForward):
        self._move_along_local_axis(-action.distance, 2)

    def actuate_move_tangentially(self, action: MoveTangentially):
        if action.distance == 0.0:
            return
        direction = np.array(action.direction)
        direction_length = np.linalg.norm(direction)
        if np.isclose(direction_length, 0.0):
            return
        direction = direction / direction_length

        rotation = rotation_from_quat(self.rotation)
        direction_rel_world = rotation.apply(direction)
        self.position = np.array(self.position) + direction_rel_world * action.distance

    def actuate_orient_horizontal(self, action: OrientHorizontal):
        self._move_along_local_axis(-action.left_distance, 0)
        self._actuate_yaw(-action.rotation_degrees)
        self._move_along_local_axis(-action.forward_distance, 2)

    def actuate_orient_vertical(self, action: OrientVertical):
        self._move_along_local_axis(-action.down_distance, 1)
        self._actuate_pitch(-action.rotation_degrees)
        self._move_along_local_axis(-action.forward_distance, 2)
