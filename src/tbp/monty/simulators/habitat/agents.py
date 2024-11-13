# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT..

import uuid
from collections import defaultdict
from typing import List, Tuple

import habitat_sim
import numpy as np
import quaternion as qt
from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration, AgentState

from .sensors import RGBDSensorConfig, SemanticSensorConfig, SensorConfig

__all__ = [
    "HabitatAgent",
    "SingleSensorAgent",
    "MultiSensorAgent",
]

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Size = Tuple[int, int]


class HabitatAgent:
    """Habitat agent wrapper.

    Agents are used to define moveable bodies in the environment.
    Every habitat agent will inherit from this class.

    Attributes:
        agent_id: Unique ID of this agent in env. Observations returned by environment
            will be mapped to this id. ``{"agent_id": {"sensor": [...]}}``. Actions
            provided by this sensor module will be prefixed by this id. i.e.
            "agent_id.move_forward"
        position: Module initial position in meters. Default (0, 1.5, 0)
        rotation: Module initial rotation quaternion. Default (1, 0, 0, 0)
        height: Module height in meters. Default 0.0
    """

    def __init__(
        self,
        agent_id: str,
        position: Vector3 = (0.0, 1.5, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
    ):
        if agent_id is None:
            agent_id = uuid.uuid4().hex
        self.agent_id = agent_id
        self.position = position
        self.rotation = rotation
        self.height = height
        self.sensors: List[SensorConfig] = []

    def get_spec(self):
        """Returns a habitat-sim agent configuration.

        Returns:
            :class:`habitat_sim.agent.AgentConfiguration` spec create from this
            sensor module configuration.
        """
        spec = AgentConfiguration()
        spec.height = self.height
        for sensor in self.sensors:
            spec.sensor_specifications.extend(sensor.get_specs())
        return spec

    def initialize(self, simulator):
        """Initialize habitat-sim agent runtime state.

        This method must be called to update the agent and sensors runtime
        instance. This is necessary because some of the configuration attributes
        requires access to the instanciated node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        # Initialize agent state
        agent_state = AgentState()
        agent_state.position = self.position
        rotation = np.quaternion(*self.rotation)
        agent_state.rotation = rotation
        simulator.initialize_agent(self.agent_id, agent_state)

    def process_observations(self, agent_obs):
        """Callback processing raw habitat agent observations to Monty-compatible ones.

        Args:
            agent_obs: Agent raw habitat-sim observations

        Returns:
            dict: The processed observations grouped by sensor_id
        """
        # Habitat raw sensor observations are flat where the observation key is
        # composed of the `sensor_id.sensor_type`. The default agent starts by
        # grouping habitat raw observation by sensor_id and sensor_type.
        obs_by_sensor = defaultdict(dict)
        for sensor_key, data in agent_obs.items():
            sensor_id, sensor_type = sensor_key.split(".")
            obs_by_sensor[sensor_id][sensor_type] = data

        # Call each sensor to postprocess the observation data
        for sensor in self.sensors:
            sensor_id = sensor.sensor_id
            sensor_obs = obs_by_sensor.get(sensor_id)
            if sensor_obs is not None:
                obs_by_sensor[sensor_id] = sensor.process_observations(sensor_obs)

        return obs_by_sensor


class ActionSpaceMixin:
    """An auxiliary function for agent classes to return their action space."""

    def get_action_space(self, spec):
        """Creates and returns the agent's action space dictionary.

        Action space can be absolute, for distant-agent, or for surface-agent
        This method is only used in a couple of unit tests at the moment
        If not, use a default action space.
        action spaces are formatted as Tuple of lists, with elements:
            0: action name
            1: (initial) action amount
            2: (initial) constraint

        Args:
            spec (AgentConfiguration): agent parameters

        Returns:
            AgentConfiguration: agent parameters updated with action space
        """
        absolute_only_action_space = (
            ["set_yaw", 0.0, None],
            ["set_agent_pitch", 0.0, None],
            ["set_sensor_pitch", 0.0, None],
            ["set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None],
            ["set_sensor_rotation", [[qt.one]], None],
            ["set_sensor_pose", [[0.0, 0.0, 0.0], qt.one], None],
            # TODO triple check qt.one is correct format; expects numpy, so
            # should be fine
        )
        distant_agent_action_space = (
            ["move_forward", self.translation_step, None],
            ["turn_left", self.rotation_step, None],
            ["turn_right", self.rotation_step, None],
            ["look_up", self.rotation_step, 90.0],
            ["look_down", self.rotation_step, 90.0],
            ["set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None],
            ["set_sensor_rotation", [[qt.one]], None],
        )
        surface_agent_action_space = (
            ["move_forward", self.translation_step, None],
            ["move_tangentially", self.translation_step, None],
            ["orient_horizontal", self.rotation_step, None],
            ["orient_vertical", self.rotation_step, None],
            ["set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None],
            ["set_sensor_rotation", [[qt.one]], None],
        )
        if self.action_space_type == "absolute_only":
            action_space = absolute_only_action_space
        elif self.action_space_type == "distant_agent":
            action_space = distant_agent_action_space
        elif self.action_space_type == "surface_agent":
            action_space = surface_agent_action_space

        spec.action_space = {}
        for action in action_space:
            spec.action_space[f"{self.agent_id}.{action[0]}"] = ActionSpec(
                f"{action[0]}",
                ActuationSpec(amount=action[1], constraint=action[2]),
            )

        return spec


class MultiSensorAgent(HabitatAgent, ActionSpaceMixin):
    """Minimal version of a HabitatAgent with multiple RGBD sensors mounted.

    The RGBD sensors are mounted to the same movable object (like two go-pros
    mounted to a helmet) with the following pre-defined actions:

        - "`agent_id`.move_forward": Move camera forward using `translation_step`
        - "`agent_id`.turn_left": Turn camera left `rotation_step`
        - "`agent_id`.turn_right": Turn camera right `rotation_step`
        - "`agent_id`.look_up": Turn the camera up `rotation_step`
        - "`agent_id`.look_down": Turn the camera down `rotation_step`
        - "`agent_id`".set_yaw" : Set camera agent absolute yaw value
        - "`agent_id`".set_sensor_pitch" : Set camera sensor absolute pitch value
        - "`agent_id`".set_agent_pitch" : Set camera agent absolute pitch value

    Each camera will return the following observations:

        - "sensor_ids[i].rgba": Color information for every pixel (x, y, 4)
        - "sensor_ids[i].depth": Depth information for every pixel (x, y, 1)
        - "sensor_ids[i].semantic": Optional object semantic information for every pixel
                                    (x, y, 1)

        where i is an integer indexing the list of sensor_ids.

    Note:
        The parameters `resolutions`, `rotations` and so on effectively specify
        both the number of sensors, and the sensor parameters. For N sensors,
        specify a list of N `resolutions`, and so on. All lists must be the same
        length. By default, a list of length one will be provided. Therefore, do
        not leave an argument blank if you wish to run a simulation with N > 1
        sensors.

    Note:
        The parameters `translation_step` and `rotation_step` are set to 0 by
        default. All action amounts should be specified by the MotorSystem.

    Attributes:
        agent_id: Actions provided by this camera will be prefixed by this id.
            Default "camera"
        sensor_ids: List of ids for each sensor. Actions are prefixed with agent id,
            but observations are prefixed with sensor id.
        resolutions: List of camera resolutions (width, height). Defaut = (16, 16)
        positions: List of camera initial absolute positions in meters, relative
            to the agent.
        rotations: List of camera rotations (quaternion). Default (1, 0, 0, 0)
        zooms: List of camera zoom multipliers. Use >1 to increase, 0<factor<1 to
            decrease. Default 1.0
        semantics: List of booleans deciding if each RGBD sensor also gets a semantic
            sensor with it. Default = False
        height: Height of the mount itself in meters. Default 0.0 (but position of
            the agent will be 1.5 meters in "height" dimension)
        rotation_step: Rotation step in degrees used by the `turn_*` and
            `look_*` actions. Default 0 degrees
        translation_step: Translation step is meters used by the `move_*` actions.
            Default: 0m
        action_space_type: Decides between three action spaces.
            "distant_agent" actions saccade like a ball-in-socket joint, viewing an
            object from a distance "surface_agent" actions orient to an object surface
            and move tangentially along it "absolute_only" actions are movements in
            absolute world coordinates only
    """

    def __init__(
        self,
        agent_id: str,
        sensor_ids: Tuple[str],
        position: Vector3 = (0.0, 1.5, 0.0),  # Agent position
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
        rotation_step: float = 0.0,
        translation_step: float = 0.0,
        action_space_type: str = "distant_agent",
        resolutions: Tuple[Size] = ((16, 16),),
        positions: Tuple[Vector3] = ((0.0, 0.0, 0.0),),
        rotations: Tuple[Quaternion] = ((1.0, 0.0, 0.0, 0.0),),
        zooms: Tuple[float] = (1.0,),
        semantics: Tuple[bool] = (False,),
    ):
        super().__init__(agent_id, position, rotation, height)
        if sensor_ids is None:
            sensor_ids = (uuid.uuid4().hex,)
        self.sensor_ids = sensor_ids
        self.rotation_step = rotation_step
        self.translation_step = translation_step
        self.action_space_type = action_space_type
        self.resolutions = resolutions
        self.positions = positions
        self.rotations = rotations
        self.zooms = zooms
        self.semantics = semantics

        param_lists = [
            self.sensor_ids,
            self.resolutions,
            self.positions,
            self.rotations,
            self.zooms,
            self.semantics,
        ]
        num_sensors = len(self.sensor_ids)
        assert all(len(p) == num_sensors for p in param_lists)

        for sid, res, pos, rot, zoom, sem in zip(*param_lists):
            # Add RGBD Camera
            sensor_position = (pos[0], pos[1] + self.height, pos[2])
            rgbd_sensor = RGBDSensorConfig(
                sensor_id=sid,
                position=sensor_position,
                rotation=rot,
                resolution=res,
                zoom=zoom,
            )
            self.sensors.append(rgbd_sensor)

            # Add optional semantic sensor
            if sem:
                semantic_sensor = SemanticSensorConfig(
                    sensor_id=sid,
                    position=sensor_position,
                    rotation=rot,
                    resolution=res,
                    zoom=zoom,
                )
                self.sensors.append(semantic_sensor)

    def get_spec(self):
        spec = super().get_spec()
        spec = self.get_action_space(spec)
        return spec

    def initialize(self, simulator):
        """Initialize agent runtime state.

        This method must be called by :class:`.HabitatSim` to update the agent
        and sensors runtime instance. This is necessary because some of the
        configuration attributes requires access to the instanciated scene node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        super().initialize(simulator)

        # Initialze camera zoom
        agent = simulator.get_agent(self.agent_id)
        for i, sensor_id in enumerate(self.sensor_ids):
            zoom = self.zooms[i]
            sensor_types = ["rgba", "depth"]
            if self.semantics[i]:
                sensor_types.append("semantic")
            for s in sensor_types:
                # FIXME: Using protected member `_sensor`
                camera = agent._sensors[f"{sensor_id}.{s}"]
                camera.zoom(zoom)


class SingleSensorAgent(HabitatAgent, ActionSpaceMixin):
    """Minimal version of a HabitatAgent.

    This is the special case of :class:`MultiSensorAgent` when there is at most 1
    RGBD and 1 semantic sensor. Thus, the arguments are single values instead of
    lists.
    """

    def __init__(
        self,
        agent_id: str,
        sensor_id: str,
        agent_position: Vector3 = (0.0, 1.5, 0.0),
        sensor_position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
        resolution: Size = (16, 16),
        zoom: float = 1.0,
        semantic: bool = False,
        rotation_step: float = 0.0,
        translation_step: float = 0.0,
        action_space: Tuple = None,
        action_space_type: str = "distant_agent",
    ):
        """Initialize agent runtime state.

        Note that like the multi-sensor agent, the position of the agent is treated
        separately from the position of the sensor (which is relative to the agent).
        """
        super().__init__(agent_id, agent_position, rotation, height)

        if sensor_id is None:
            sensor_id = uuid.uuid4().hex
        self.sensor_id = sensor_id
        self.sensor_position = sensor_position
        self.resolution = resolution
        self.zoom = zoom
        self.semantic = semantic
        self.rotation_step = rotation_step
        self.translation_step = translation_step
        self.action_space = action_space
        self.action_space_type = action_space_type

        # Add RGBD Camera
        effective_sensor_position = (
            self.sensor_position[0],
            self.sensor_position[1] + self.height,
            self.sensor_position[2],
        )
        rgbd_sensor = RGBDSensorConfig(
            sensor_id=self.sensor_id,
            position=effective_sensor_position,
            rotation=self.rotation,
            resolution=self.resolution,
            zoom=self.zoom,
        )
        self.sensors.append(rgbd_sensor)

        # Add optional semantic sensor
        if self.semantic:
            semantic_sensor = SemanticSensorConfig(
                sensor_id=self.sensor_id,
                position=effective_sensor_position,
                rotation=self.rotation,
                resolution=self.resolution,
                zoom=self.zoom,
            )
            self.sensors.append(semantic_sensor)

    def get_spec(self):
        spec = super().get_spec()
        spec = self.get_action_space(spec)
        return spec

    def initialize(self, simulator):
        """Initialize agent runtime state.

        This method must be called by :class:`.HabitatSim` to update the agent
        and sensors runtime instance. This is necessary because some of the
        configuration attributes requires access to the instanciated scene node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        super().initialize(simulator)

        # Update camera zoom only when Zoom value is different than 1x
        if self.zoom != 1.0:
            agent = simulator.get_agent(self.agent_id)
            agent_config = agent.agent_config
            for sensor in agent_config.sensor_specifications:
                if isinstance(sensor, habitat_sim.CameraSensorSpec):
                    # FIXME: Using protected member `_sensor`
                    camera = agent._sensors[sensor.uuid]
                    camera.zoom(self.zoom)
