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
from typing import TYPE_CHECKING, cast

import numpy as np
import quaternion as qt
from mujoco import MjsBody, mjtJoint

from tbp.monty.frameworks.agents import Agent, AgentID
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
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
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
        # to be indexed into to pull out the relevant values.
        # TODO: do we need to repeatedly look this up?
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
        for sensor_id in self._sensor_configs:
            renderer = self.sim.renderer
            renderer.update_scene(self.sim.data, camera=f"{self.id}.{sensor_id}")
            rgba_data = renderer.render()

            renderer.enable_depth_rendering()
            # TODO: do we need to do this between rendering?
            renderer.update_scene(self.sim.data, camera=f"{self.id}.{sensor_id}")
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
