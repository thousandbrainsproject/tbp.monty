# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import uuid
from typing import NewType, Optional

import mujoco
from mujoco import MjData, MjModel, MjsBody, MjSpec

from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)

AgentID = NewType("AgentID", str)
SensorID = NewType("SensorID", str)


class MuJoCoAgent:
    """A MuJoCo equivalent of a HabitatAgent.

    MuJoCo doesn't have a concept of an Agent like HabitatSim does, so this
    class provides an abstraction around MuJoCo's bodies and sensors to provide
    a similar concept.
    """

    def __init__(
        self,
        agent_id: Optional[AgentID],
        position: VectorXYZ = (0.0, 1.5, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
    ):
        """MuJoCoAgent constructor.

        Args:
            agent_id: Unique ID for this agent in the environment. Observations will be
                mapped to this ID.
            position: Initial position.
            rotation: Initial rotation quaternion.
        """
        if agent_id is None:
            agent_id = AgentID(uuid.uuid4().hex)
        self.agent_id = agent_id
        self.agent_name = f"agent{self.agent_id}"
        self._initial_position = position
        self._initial_rotation = rotation

    def initialize(self, spec: MjSpec) -> None:
        """Initialize the agent in the simulator.

        This method is used by the simulator to initialize the agent within
        the environment. Agents are responsible for creating whatever structures
        they need within the simulator spec in order to function.
        """

    def observe(self, model: MjModel, data: MjData):
        """Return an observation for this agent's sensors."""


class SingleSensorAgent(MuJoCoAgent):
    """A simple agent with a single camera sensor."""

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID,
        position: VectorXYZ = (0.0, 1.5, 0.0),
        sensor_position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
    ):
        """SingleSensorAgent constructor.

        Args:
            agent_id: Unique ID for this agent in the environment.
            sensor_id: Unique ID for the camera sensor.
            position: Initial position.
            sensor_position: Initial position of the sensor, relative to the
                agent body.
            rotation: Initial rotation quaternion.
        """
        super().__init__(agent_id, position, rotation)
        if sensor_id is None:
            sensor_id = SensorID(uuid.uuid4().hex)
        self.sensor_id = sensor_id
        self.sensor_name = f"sensor{self.sensor_id}"
        self._initial_sensor_position = sensor_position
        self._resolution = (64, 64)

    def initialize(self, spec: MjSpec) -> None:
        """Initialize the agent in the simulator.

        Creates a body in the world and nests the single sensor inside of it.
        """
        worldbody: MjsBody = spec.worldbody
        body: MjsBody = worldbody.add_body(
            name=f"{self.agent_name}_body",
            pos=self._initial_position,
        )
        body.add_camera(
            resolution=self._resolution,
            name=self.sensor_name,
            pos=self._initial_sensor_position,
            # Single sensor uses agent orientation
            quat=self._initial_rotation,
        )

    def observe(self, model: MjModel, data: MjData):
        with mujoco.Renderer(
            model, width=self._resolution[0], height=self._resolution[1]
        ) as renderer:
            # Render RGB data
            renderer.update_scene(data, camera=self.sensor_name)
            rgb_data = renderer.render()
            # Render Depth data
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=self.sensor_name)
            depth_data = renderer.render()
            # Render semantic data
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=self.sensor_name)
            semantic_data = renderer.render()

            return {
                self.sensor_id: {
                    "rgb": rgb_data,
                    "depth": depth_data,
                    "semantic": semantic_data[:, :, 0],
                }
            }
