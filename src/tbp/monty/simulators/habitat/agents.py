# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Tuple

import habitat_sim
import quaternion as qt
from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration, AgentState
from typing_extensions import Literal

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.habitat.sensors import (
    RGBDSensorConfig,
    SemanticSensorConfig,
    SensorConfig,
)

__all__ = [
    "HabitatAgent",
    "MultiSensorAgent",
    "SingleSensorAgent",
]

ActionSpaceName = Literal["absolute_only", "distant_agent", "surface_agent"]
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Size = Tuple[int, int]


class HabitatAgent:
    """Habitat agent wrapper.

    Agents are used to define movable bodies in the environment.
    Every Habitat agent will inherit from this class.

    Attributes:
        agent_id: Unique ID of this agent in the environment.
            Observations returned by the environment will be mapped to this ID.
            ``{"agent_id": {"sensor": [...]}}``.
            Actions provided by this sensor module will be prefixed by
            this ID, i.e. "agent_id.move_forward"
        position: Module initial position in meters. Defaults to (0, 1.5, 0).
        rotation: Module initial rotation quaternion. Defaults to (1, 0, 0, 0).
        height: Module height in meters. Defaults to 0.0.
    """

    def __init__(
        self,
        agent_id: AgentID | None,
        position: Vector3 = (0.0, 1.5, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
    ):
        if agent_id is None:
            agent_id = AgentID(uuid.uuid4().hex)
        self.agent_id = agent_id
        self.position = position
        self.rotation = rotation
        self.height = height
        self.sensors: list[SensorConfig] = []
        self.habitat_to_monty_sensor_id_map: dict[str, str] = {}

    def get_spec(self) -> AgentConfiguration:
        """Return a habitat-sim agent configuration.

        Returns:
            Spec created from this sensor module configuration.
        """
        spec = AgentConfiguration()
        spec.height = self.height
        self.habitat_to_monty_sensor_id_map.clear()
        for sensor in self.sensors:
            sensor_specs = sensor.get_specs()
            for sensor_spec in sensor_specs:
                habitat_id = sensor_spec.uuid
                monty_id = sensor.sensor_id
                self.habitat_to_monty_sensor_id_map[habitat_id] = monty_id
            spec.sensor_specifications.extend(sensor_specs)
        return spec

    def initialize(self, simulator):
        """Initialize the habitat-sim agent's runtime state.

        This method must be called to update the agent and sensors runtime
        instance because some of the configuration attributes require access to
        the instantiated node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        # Initialize agent state
        agent_state = AgentState()
        agent_state.position = self.position
        rotation = qt.quaternion(*self.rotation)
        agent_state.rotation = rotation
        simulator.initialize_agent(self.agent_id, agent_state)

    def process_observations(self, agent_obs) -> AgentObservations:
        """Callback that processes raw Habitat agent observations.

        Converts raw observations into Monty-compatible ones.

        Args:
            agent_obs: Raw habitat-sim observations from the agent

        Returns:
            The processed observations grouped by sensor_id
        """
        # Habitat raw sensor observations are flat, where the observation key is
        # composed of the `sensor_id.sensor_type`. The default agent starts by
        # grouping habitat raw observations by sensor_id and sensor_type.
        obs_by_sensor: AgentObservations = defaultdict(dict)
        for sensor_key, data in agent_obs.items():
            sensor_id, sensor_type = sensor_key.split(".")
            obs_by_sensor[SensorID(sensor_id)][sensor_type] = data

        # Call each sensor to post-process the observation data
        for sensor in self.sensors:
            sensor_id = SensorID(sensor.sensor_id)
            sensor_obs = obs_by_sensor.get(sensor_id)
            if sensor_obs is not None:
                obs_by_sensor[sensor_id] = sensor.process_observations(sensor_obs)

        return obs_by_sensor


def action_space(
    action_space_type: ActionSpaceName,
    agent_id: str,
    translation_step: float,
    rotation_step: float,
) -> dict[str, ActionSpec]:
    """Generate an action space for a given action space type.

    Action space can be `absolute_only`, `distant_agent`, or `surface_agent`.
    This method is currently used only in a couple of unit tests; otherwise,
    use a default action space.
    Action spaces are formatted as lists of tuples, with elements:
        0: action name
        1: (initial) action amount
        2: (initial) constraint

    Args:
        action_space_type: The type of action space to generate.
        agent_id: The ID of the agent.
        translation_step: The translation step.
        rotation_step: The rotation step.

    Returns:
        The generated action space.
    """
    absolute_only_action_space: list[
        tuple[str, float | list[float | qt.quaternion], float | None]
    ] = [
        ("set_yaw", 0.0, None),
        ("set_agent_pitch", 0.0, None),
        ("set_sensor_pitch", 0.0, None),
        ("set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None),
        ("set_sensor_rotation", [[qt.one]], None),
        ("set_sensor_pose", [[0.0, 0.0, 0.0], qt.one], None),
    ]
    distant_agent_action_space: list[
        tuple[str, float | list[float | qt.quaternion], float | None]
    ] = [
        ("move_forward", translation_step, None),
        ("turn_left", rotation_step, None),
        ("turn_right", rotation_step, None),
        ("look_up", rotation_step, 90.0),
        ("look_down", rotation_step, 90.0),
        ("set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None),
        ("set_sensor_rotation", [[qt.one]], None),
    ]
    surface_agent_action_space: list[
        tuple[str, float | list[float | qt.quaternion], float | None]
    ] = [
        ("move_forward", translation_step, None),
        ("move_tangentially", translation_step, None),
        ("orient_horizontal", rotation_step, None),
        ("orient_vertical", rotation_step, None),
        ("set_agent_pose", [[0.0, 0.0, 0.0], qt.one], None),
        ("set_sensor_rotation", [[qt.one]], None),
    ]
    if action_space_type == "absolute_only":
        action_spec = absolute_only_action_space
    elif action_space_type == "distant_agent":
        action_spec = distant_agent_action_space
    elif action_space_type == "surface_agent":
        action_spec = surface_agent_action_space

    action_space: dict[str, ActionSpec] = {}
    for action in action_spec:
        action_space[f"{agent_id}.{action[0]}"] = ActionSpec(
            f"{action[0]}",
            ActuationSpec(
                amount=action[1],
                constraint=action[2],
            ),  # type: ignore[arg-type]
        )

    return action_space


class MultiSensorAgent(HabitatAgent):
    """Minimal version of a HabitatAgent with multiple RGBD sensors mounted.

    The RGBD sensors are mounted to the same movable object (like two go-pros
    mounted to a helmet) with the following pre-defined actions:

        - "`agent_id`.move_forward": Move camera forward using `translation_step`
        - "`agent_id`.turn_left": Turn camera left `rotation_step`
        - "`agent_id`.turn_right": Turn camera right `rotation_step`
        - "`agent_id`.look_up": Turn the camera up `rotation_step`
        - "`agent_id`.look_down": Turn the camera down `rotation_step`
        - "`agent_id`.set_yaw": Set the camera agent's absolute yaw value
        - "`agent_id`.set_sensor_pitch": Set the camera sensor's absolute pitch value
        - "`agent_id`.set_agent_pitch": Set the camera agent's absolute pitch value

    Each camera will return the following observations:

        - "sensor_ids[i].rgba": Color information for every pixel (x, y, 4)
        - "sensor_ids[i].depth": Depth information for every pixel (x, y, 1)
        - "sensor_ids[i].semantic": Optional object semantic information for every pixel
                                    (x, y, 1)

        where i is an integer indexing the list of sensor_ids.

    Note:
        The parameters `resolutions`, `rotations`, and so on effectively specify
        both the number of sensors, and the sensor parameters. For N sensors,
        specify a list of N `resolutions`, and so on. All lists must be the same
        length. By default, a list of length one will be provided. Therefore, do
        not leave an argument blank if you wish to run a simulation with N > 1
        sensors.

    Note:
        The parameters `translation_step` and `rotation_step` are set to 0 by
        default. All action amounts should be specified by the MotorSystem.

    Attributes:
        agent_id: Actions provided by this camera will be prefixed by this ID.
            Defaults to "camera".
        sensor_ids: List of IDs for each sensor. Actions are prefixed with agent ID,
            but observations are prefixed with sensor ID.
        resolutions: List of camera resolutions (width, height). Defaults to (16, 16).
        positions: List of camera initial absolute positions in meters, relative
            to the agent.
        rotations: List of camera rotations (quaternion). Defaults to (1, 0, 0, 0).
        zooms: List of camera zoom multipliers. Use >1 to increase and 0 < factor < 1
            to decrease. Defaults to 1.0.
        semantics: List of booleans determining if each RGBD sensor also gets a semantic
            sensor with it. Defaults to False.
        height: Height of the mount itself in meters. Defaults to 0.0 (but position of
            the agent will be 1.5 meters in the "height" dimension).
        rotation_step: Rotation step in degrees used by the `turn_*` and
            `look_*` actions. Defaults to 0 degrees.
        translation_step: Translation length in meters used by the `move_*` actions.
            Defaults to 0 m.
        action_space_type: Decides between three action spaces:
            "distant_agent" actions saccade like a ball-in-socket joint, viewing an
            object from a distance.
            "surface_agent" actions orient to an object surface and move tangentially
            along it.
            "absolute_only" actions are movements in absolute world coordinates only.
    """

    def __init__(
        self,
        agent_id: AgentID | None,
        sensor_ids: tuple[str],
        position: Vector3 = (0.0, 1.5, 0.0),  # Agent position
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
        rotation_step: float = 0.0,
        translation_step: float = 0.0,
        action_space_type: ActionSpaceName = "distant_agent",
        resolutions: tuple[Size] = ((16, 16),),
        positions: tuple[Vector3] = ((0.0, 0.0, 0.0),),
        rotations: tuple[Quaternion] = ((1.0, 0.0, 0.0, 0.0),),
        zooms: tuple[float] = (1.0,),
        semantics: tuple[bool] = (False,),
    ):
        super().__init__(agent_id, position, rotation, height)
        if sensor_ids is None:
            sensor_ids = (uuid.uuid4().hex,)
        self.sensor_ids = sensor_ids
        self.rotation_step = rotation_step
        self.translation_step = translation_step
        self.action_space_type: ActionSpaceName = action_space_type
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
        spec.action_space = action_space(
            self.action_space_type,
            self.agent_id,
            self.translation_step,
            self.rotation_step,
        )
        return spec

    def initialize(self, simulator):
        """Initialize agent runtime state.

        This method must be called by :class:`.HabitatSim` to update the agent
        and sensors runtime instance. This is necessary because some of the
        configuration attributes require access to the instantiated scene node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        super().initialize(simulator)

        # Initialize camera zoom
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


class SingleSensorAgent(HabitatAgent):
    """Minimal version of a HabitatAgent.

    This is the special case of :class:`MultiSensorAgent` when there is at most 1
    RGBD and 1 semantic sensor. Thus, the arguments are single values instead of
    lists.
    """

    def __init__(
        self,
        agent_id: AgentID | None,
        sensor_id: SensorID,
        agent_position: Vector3 = (0.0, 1.5, 0.0),
        sensor_position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        height: float = 0.0,
        resolution: Size = (16, 16),
        zoom: float = 1.0,
        semantic: bool = False,
        rotation_step: float = 0.0,
        translation_step: float = 0.0,
        action_space_type: ActionSpaceName = "distant_agent",
    ):
        """Initialize agent runtime state.

        Note that, like the multi-sensor agent, the position of the agent is treated
        separately from the position of the sensor (which is relative to the agent).
        """
        super().__init__(agent_id, agent_position, rotation, height)

        self.sensor_id = sensor_id
        self.sensor_position = sensor_position
        self.resolution = resolution
        self.zoom = zoom
        self.semantic = semantic
        self.rotation_step = rotation_step
        self.translation_step = translation_step
        self.action_space_type: ActionSpaceName = action_space_type

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
        spec.action_space = action_space(
            self.action_space_type,
            self.agent_id,
            self.translation_step,
            self.rotation_step,
        )
        return spec

    def initialize(self, simulator):
        """Initialize agent runtime state.

        This method must be called by :class:`.HabitatSim` to update the agent
        and sensors runtime instance. This is necessary because some of the
        configuration attributes require access to the instantiated scene node.

        Args:
            simulator: Instantiated :class:`.HabitatSim` instance
        """
        super().initialize(simulator)

        # Update camera zoom only when the zoom value differs from 1x
        if self.zoom != 1.0:
            agent = simulator.get_agent(self.agent_id)
            agent_config = agent.agent_config
            for sensor in agent_config.sensor_specifications:
                if isinstance(sensor, habitat_sim.CameraSensorSpec):
                    # FIXME: Using protected member `_sensor`
                    camera = agent._sensors[sensor.uuid]
                    camera.zoom(self.zoom)
