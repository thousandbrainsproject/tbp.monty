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
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import quaternion as qt
from mujoco import MjsBody, mjtJoint
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
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
from tbp.monty.frameworks.sensors import SensorConfig, SensorID
from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
    rotation_from_quat,
)
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ

if TYPE_CHECKING:
    from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator


logger = logging.getLogger(__name__)

# The default field of view value for zoom 1.0
# Note: this value is the half-FOV rather than the full FOV
DEFAULT_CAMERA_FOVY: float = 45.0


class Axis(IntEnum):
    """Axis for the purposes of local movement."""

    # Values map to indices in a rotation matrix
    X = 0
    Y = 1
    Z = 2


class Agent(Protocol):
    """Protocol for an agent that interacts with an environment."""

    id: AgentID

    @property
    def observations(self) -> AgentObservations:
        """Returns the current observations of the sensors coupled to this agent."""

    @property
    def state(self) -> AgentState:
        """Returns the current proprioceptive state of the agent."""

    def reset(self) -> None:
        """Resets the agent to its initial state."""


class Embodiment(Agent):
    """The embodiment of an agent inside the simulator.

    These are responsible for positioning a collection of sensors, moving and
    reorienting them in the environment, and returning observations and
    proprioceptive state.

    To create an agent that responds to various Actions, create a class that
    contains an instance of Embodiment, and have it interact with its Embodiment
    to affect the environment.
    """

    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_configs: dict[SensorID, SensorConfig],
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
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
            # Needed to use joints
            mass=1.0,
            inertia=(1.0, 1.0, 1.0),
        )
        self.agent_joint = agent_body.add_freejoint()

        self.sensor_body_id = f"{agent_id}.sensor"
        sensor_body: MjsBody = agent_body.add_body(
            name=self.sensor_body_id,
            pos=ZERO_VECTOR,
            quat=IDENTITY_QUATERNION,
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
        # MuJoCo stores coordinates in an array-like structure that has
        # to be indexed into to pull out the relevant values. Because the
        # `sim.model` could change due to a recompile in the simulator,
        # e.g. after adding a new object to the scene, we need to look up
        # the "address" each time.
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
            renderer = self.sim.renderer_for_res(sensor_cfg["resolution"])
            renderer.update_scene(self.sim.data, camera=f"{self.id}.{sensor_id}")
            rgba_data = renderer.render()

            renderer.enable_depth_rendering()
            depth_data = renderer.render()
            renderer.disable_depth_rendering()

            obs[sensor_id] = SensorObservation(
                depth=depth_data,
                rgba=rgba_data,
            )
        return obs

    @property
    def state(self) -> AgentState:
        # Calculate sensor position and rotation relative to the agent.
        # Rotation is shared since it's from the sensor body containing all the
        # sensors, while individual sensor positions are calculated separately below.
        # Note: the sensor body position and rotation is returned relative to world
        # coordinates from the simulator.
        sensor_body_rot = rotation_from_quat(
            self.sim.data.body(self.sensor_body_id).xquat
        )
        agent_rotation = rotation_from_quat(self.rotation)
        sensor_body_rot_rel_agent = agent_rotation.inv() * sensor_body_rot
        sensor_body_rot_quat = qt.quaternion(
            *rotation_as_quat(sensor_body_rot_rel_agent)
        )

        sensor_states = {}
        for sensor_id, sensor_cfg in self._sensor_configs.items():
            # This code assumes that the position of the sensor body, or sensors
            # relative to the agent CANNOT change. This is because we use the
            # configured position of the sensor (rel. agent) to compute positions.
            # This constraint can be removed by computing the sensor's position
            # relative agent from their world coordinates.
            sensor_pos_rel_agent = sensor_body_rot_rel_agent.apply(
                sensor_cfg["position"]
            )
            sensor_states[sensor_id] = SensorState(
                position=cast("VectorXYZ", tuple(sensor_pos_rel_agent)),
                rotation=sensor_body_rot_quat,
            )
        return AgentState(
            position=self.position,
            rotation=qt.quaternion(*self.rotation),
            sensors=sensor_states,
        )

    def reset(self) -> None:
        self.position = self._initial_position
        self.rotation = self._initial_rotation
        self.set_sensor_rotation(IDENTITY_QUATERNION)

    def move_along_local_axis(self, distance: float, axis: Axis) -> None:
        """Move the embodiment along an axis relative to its local basis."""
        rotation = rotation_from_quat(self.rotation)
        rotation_matrix = rotation.as_matrix()
        axis_vector = rotation_matrix[:, axis] * distance
        new_xyz = np.array(self.position) + axis_vector
        self.position = new_xyz

    def yaw(self, delta_theta: float) -> None:
        """Yaw the embodiment by delta_theta degrees."""
        delta_theta_rot = Rotation.from_euler("xyz", (0, delta_theta, 0), degrees=True)
        rotation = rotation_from_quat(self.rotation)
        new_rotation = rotation * delta_theta_rot
        self.rotation = rotation_as_quat(new_rotation)

    def pitch(self, delta_phi: float) -> None:
        """Pitch the sensor body by delta_phi degrees.

        Note: this DOES NOT change the orientation of the embodiment.
        """
        delta_phi = np.deg2rad(delta_phi)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] += delta_phi

    def set_pose(self, location: VectorXYZ, rotation: QuaternionWXYZ) -> None:
        """Set the location and rotation of the embodiment to the provided values."""
        # TODO: replace with a property setter using a Pose object
        self.position = location
        self.rotation = np.array(rotation)

    def set_sensor_rotation(self, rotation_quat: QuaternionWXYZ) -> None:
        """Sets the orientation of the sensor body, relative to the embodiment.

        Note: while this does take a full rotation quaternion, it DOES NOT change the
        orientation outside the x-axis of the sensor body, i.e. the body can only
        pitch up and down.
        """
        rotation = rotation_from_quat(rotation_quat)
        angles = rotation.as_euler("xyz", degrees=False)
        qpos_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[qpos_addr] = angles[0]


class NoopAgent(Agent):
    """A simple multi-sensor agent that doesn't respond to actions.

    It does not implement any of the actuate methods defined by the various
    Action Actuators. The simulator is designed to catch the errors for these
    missing methods and log that the agent doesn't understand them.

    It also cannot be used with a positioning procedure, since it can't move,
    and the procedure will make no forward progress.
    """

    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_configs: dict[SensorID, SensorConfig],
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
    ):
        self._embodiment = Embodiment(
            simulator, agent_id, sensor_configs, position, rotation
        )
        self.id = agent_id

    @property
    def state(self) -> AgentState:
        return self._embodiment.state

    @property
    def observations(self) -> AgentObservations:
        return self._embodiment.observations

    def reset(self) -> None:
        self._embodiment.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


class DistantAgent(Agent):
    """A multi-sensor agent for sensing objects from a distance."""

    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_configs: dict[SensorID, SensorConfig],
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
    ):
        self._embodiment = Embodiment(
            simulator, agent_id, sensor_configs, position, rotation
        )
        self.id = agent_id

    @property
    def state(self) -> AgentState:
        return self._embodiment.state

    @property
    def observations(self) -> AgentObservations:
        return self._embodiment.observations

    def reset(self) -> None:
        self._embodiment.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        rotation = action.rotation_quat
        if isinstance(rotation, qt.quaternion):
            # TODO: Fix all the places SetAgentPose is created
            logger.warning(
                "SetAgentPose rotation is a qt.quaternion and not a QuaternionWXYZ."
            )
            rotation = cast("QuaternionWXYZ", tuple(qt.as_float_array(rotation)))
        self._embodiment.set_pose(action.location, rotation)

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        rotation = action.rotation_quat
        if isinstance(rotation, qt.quaternion):
            # TODO: Fix all the places SetSensorRotation is created
            logger.warning(
                "SetSensorRotation rotation is a qt.quaternion and "
                "not a QuaternionWXYZ."
            )
            rotation = cast("QuaternionWXYZ", tuple(qt.as_float_array(rotation)))
        self._embodiment.set_sensor_rotation(rotation)

    def actuate_move_forward(self, action: MoveForward) -> None:
        self._embodiment.move_along_local_axis(distance=-action.distance, axis=Axis.Z)

    def actuate_turn_right(self, action: TurnRight) -> None:
        self._embodiment.yaw(-action.rotation_degrees)

    def actuate_turn_left(self, action: TurnLeft) -> None:
        self._embodiment.yaw(action.rotation_degrees)

    def actuate_look_up(self, action: LookUp) -> None:
        self._embodiment.pitch(action.rotation_degrees)

    def actuate_look_down(self, action: LookDown) -> None:
        self._embodiment.pitch(-action.rotation_degrees)
