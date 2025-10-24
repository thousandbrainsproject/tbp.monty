# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""AI Habitat simulator interface for Monty.

See Also:
    https://github.com/facebookresearch/habitat-sim
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils import common as sim_utils
from importlib_resources import files

import tbp.monty.simulators.resources as resources
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.simulators.habitat.actuator import HabitatActuator
from tbp.monty.simulators.habitat.environment_utils import get_bounding_corners

from .agents import HabitatAgent

__all__ = [
    "HabitatSim",
    "PRIMITIVE_OBJECT_TYPES",
]

from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)

DEFAULT_SCENE = "NONE"
DEFAULT_PHYSICS_CONFIG = str(files(resources) / "default.physics_config.json")

#: Maps habitat-sim pre-configure primitive object types to semantic IDs
PRIMITIVE_OBJECT_TYPES = {
    "capsule3DSolid": 101,
    "coneSolid": 102,
    "cubeSolid": 103,
    "cylinderSolid": 104,
    "icosphereSolid": 105,
    "uvSphereSolid": 106,
}


class HabitatSim(HabitatActuator):
    """Habitat-sim interface for tbp.monty.

    This class wraps `habitat-sim <https://aihabitat.org/docs/habitat-sim>`_
    simulator for tbp.monty. It aims to hide habitat-sim internals simplifying
    experiments configuration within the tbp.monty framework.

    Example::

        camera = SingleSensorAgent(
            agent_id="camera",
            sensor_id="camera_id",
            resolution=(64, 64),
        )
        with HabitatSim(agents=[camera]) as sim:
            sim.add_object(name="coneSolid", position=(0.0, 1.5, -0.2))
            obs = sim.observations

        plot_image(obs["camera"]["camera_id"]["rgba"])
        plot_image(obs["camera"]["camera_id"]["depth"])

    Attributes:
        agents: List of :class:`HabitatAgents` to place in the simulator
        data_path: Habitat data path location, usually the same path used by
            :class:`habitat_sim.utils.environments_download`
        scene_id: Scene to use or None for empty environment.
        seed: Simulator seed to use
        enable_physics: Whether or not to enable the physics engine. Default False
    """

    def __init__(
        self,
        agents: List[HabitatAgent],
        data_path: Optional[str] = None,
        scene_id: Optional[str] = None,
        seed: int = 42,
        enable_physics: bool = True,  # Physics is now standard
    ):
        backend_config = habitat_sim.SimulatorConfiguration()
        backend_config.physics_config_file = DEFAULT_PHYSICS_CONFIG
        # NOTE that currently we do not have gravity, although this can be adjusted in
        # the above config by setting "gravity": [0, -9.8, 0]

        self.sim_enable_physics = enable_physics
        backend_config.enable_physics = self.sim_enable_physics
        backend_config.scene_id = scene_id or DEFAULT_SCENE
        backend_config.random_seed = seed

        self.np_rng = np.random.default_rng(seed)

        agent_configs = []
        self._agents = agents
        self._action_space = set()
        self._agent_id_to_index = {}

        self._object_counter = 0  # Track the number of objects added to an environment

        for index, agent in enumerate(self._agents):
            config = agent.get_spec()

            # Update global action space
            self._action_space.update(config.action_space.keys())

            # Holds a dict mapping monty's agent id to habitat agent index
            self._agent_id_to_index[agent.agent_id] = index

            agent_configs.append(config)

        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(backend_config, agent_configs)
        )

        # Load objects from data_path
        if data_path is not None:
            obj_mgr = self._sim.get_object_template_manager()
            absolute_data_path = Path(data_path).expanduser().absolute()

            v1_2_path = absolute_data_path / "configs"

            if v1_2_path.is_dir():
                # "object" sub-directory no longer exists for YCB version 1.2 (present
                # in Habitat v0.22); instead config folder is directly present
                objects_path = absolute_data_path
            else:
                # Objects downloaded with `habitat_sim.utils.environments_download` are
                # stored in the sub-dir called "objects" for older versions of YCB (eg.
                # 1.0)
                objects_path = absolute_data_path / "objects"
                # The appended /objects is also key to triggering the below -else-
                # "dataset downloaded some other way" in unit tests

            if objects_path.is_dir():
                # Search "objects" dir for habitat objects.
                # Habitat dataset objects are stored in a directory containing
                # json files with the attribures of each object in the dataset.
                # The json file name is in this format:
                # "{object_name}.object_config.json".
                # See https://aihabitat.org/docs/habitat-sim/attributesJSON.html#objectattributes # noqa: E501
                objects_data_path = {
                    f.parent for f in objects_path.glob("*/**/*.object_config.json")
                }
            else:
                # The dataset was downloaded some other way.
                # The data path must be the path to the object config files
                objects_data_path = [absolute_data_path]

            # Add each object data path to the simulator
            objects_added = False
            for path in objects_data_path:
                valid_objs = obj_mgr.load_configs(str(path), True)
                if valid_objs:
                    objects_added = True

            if not objects_added:
                self.close()
                raise ValueError(f"No valid habitat data found in {data_path}")

        for agent in self._agents:
            agent.initialize(self)

    def initialize_agent(self, agent_id, agent_state) -> None:
        """Update agent runtime state.

        Usually called first thing to update agent initial pose.

        Args:
            agent_id: Agent id of the agent to be updated
            agent_state: Agent state to update to
        """
        agent_index = self._agent_id_to_index[agent_id]
        self._sim.initialize_agent(agent_index, agent_state)

    def remove_all_objects(self) -> None:
        """Remove all objects from simulated environment."""
        rigid_mgr = self._sim.get_rigid_object_manager()
        rigid_mgr.remove_all_objects()
        self._object_counter = 0

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics=False,
        object_to_avoid=False,
        primary_target_bb=None,
    ):
        """Add new object to simulated environment.

        Args:
            name: Registered object name. It could be any of habitat-sim primitive
                objects or any configured habitat object. For a list of primitive
                objects see :const:`PRIMITIVE_OBJECT_TYPES`
            position: Object initial absolute position
            rotation: Object rotation quaternion. Default (1, 0, 0, 0)
            scale: Object scale. Default (1, 1, 1)
            semantic_id: Optional override object semantic ID
            enable_physics: Whether or not to enable physics on this objects
            object_to_avoid: If True, run collision checks to ensure the object is not
                colliding with any other objects in the scene, and otherwise move it
            primary_target_bb: If not None, the bounding box of the primary target
                object; passed when we're adding multiple objects, such that we ensure
                that the added object does not obscure the initial view of the primary
                target object (which avoiding collision alone cannot guarantee); defined
                by a list of the min and max corners

        Returns:
            RigidObject: The newly added object
        """
        obj_mgr = self._sim.get_object_template_manager()
        rigid_mgr = self._sim.get_rigid_object_manager()

        # Get first match
        obj_handle = obj_mgr.get_template_handles(name)[0]

        # Check if we are changing the object scale
        scale = tuple(scale)
        if scale != (1.0, 1.0, 1.0):
            # Get scaled object template
            scaled_obj_handle = f"{obj_handle}_scale_{scale}"
            scaled_tpl = obj_mgr.get_template_handles(scaled_obj_handle)
            if not scaled_tpl:
                # Add new template for scaled object
                scaled_tpl = obj_mgr.get_template_by_handle(obj_handle)
                scaled_tpl.scale *= mn.Vector3d(*scale)
                obj_mgr.register_template(scaled_tpl, scaled_obj_handle)
                obj_handle = scaled_obj_handle
            else:
                obj_handle = scaled_tpl[0]

        obj = rigid_mgr.add_object_by_template_handle(obj_handle)

        # Update pose
        obj.translation = position
        if isinstance(rotation, (list, tuple)):
            rotation = np.quaternion(*rotation)
        obj.rotation = sim_utils.quat_to_magnum(rotation)

        if object_to_avoid:
            assert self.sim_enable_physics, (
                "Sim-level physics must be enabled to support collision detection"
            )
            # Temporarily enable *object* physics for collision detection
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj = self.find_non_colliding_positions(
                obj,
                start_position=position,
                start_orientation=rotation,
                primary_obj_bb=primary_target_bb,
            )

        # Update object-specific physics
        # This physics setting will persist into the environment e.g. during inference
        if enable_physics:
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        else:
            # Set the motion-type to kinematic
            # (i.e. only a user-specified force/motion will affect the object), rather
            # than dynamic (i.e. object subject to forces like gravity, friction, and
            # collision detection according to physics simulations)
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # Update semantic id
        if semantic_id is not None:
            # Override default semantic id
            obj.semantic_id = semantic_id
        elif name in PRIMITIVE_OBJECT_TYPES:
            # Update semantic ID for primitive types
            obj.semantic_id = PRIMITIVE_OBJECT_TYPES[name]
        elif obj_handle in PRIMITIVE_OBJECT_TYPES:
            # Update semantic ID for primitive types
            obj.semantic_id = PRIMITIVE_OBJECT_TYPES[obj_handle]

        # Compare the intended number of objects added (counter) vs the number
        # instantiated in the Habitat environmnet
        self._object_counter += 1
        num_objects_added = self.num_objects
        if isinstance(num_objects_added, int):
            # In some units tests (e.g. MontyRunTest.test_main_with_single_experiment),
            # a simulator mock object is used, and so self.num_objects does not
            # return a meaningful int, but instead another mock object
            # TODO make this test more robust and move to its own unit test
            assert self._object_counter == num_objects_added, "Not all objects added"

        return obj

    def non_conflicting_vector(self) -> np.ndarray:
        """Find a non-conflicting vector.

        A non-conflicting vector avoids sampling directions that will be just in front
        of or behind a target object.

        Returns:
            The non-conflicting vector
        """
        angle_ranges = [
            (0, 30),
            # Forbidden 120 degrees
            (150, 180),
            (180, 210),
            # Forbidden 120 degrees
            (330, 360),
        ]

        # Choose which angle range to use
        selected_range = self.np_rng.choice(np.array(angle_ranges))
        angle_z = self.np_rng.uniform(selected_range[0], selected_range[1])

        z = np.sin(np.deg2rad(angle_z))
        x = self.np_rng.choice([-1.0, 1.0])
        return np.array([x, 0, z])

    def check_viewpoint_collision(
        self,
        primary_obj_bb,
        new_obj_bb,
        overlap_threshold=0.75,
    ) -> bool:
        """Check if the object being added overlaps in the x-axis with target.

        The object overlapping the primary target object risks obstructing the initial
        view of the agent at the start of the experiment.

        Recall that +z is out of the page, where the agent starts facing in the -z
        direction at the beginning of the episode; +y is the up vector, and +x is the
        right-ward direction

        Args:
            primary_obj_bb: the bounding box of the primary target in the scene
            new_obj_bb: the bounding box fo the new object being added
            overlap_threshold: The threshold for overlap. Defaults to 0.75.

        Returns:
            True if the overlap is greater than overlap_threshold; 1.0 corresponds
            to total overlap (the primary target is potentially not visible)
        """
        primary_start, primary_end = primary_obj_bb[0][0], primary_obj_bb[1][0]
        new_start, new_end = new_obj_bb[0][0], new_obj_bb[1][0]

        overlap_start = max(primary_start, new_start)
        overlap_end = min(primary_end, new_end)

        if overlap_start >= overlap_end:
            return False  # 0.0 percent overlap

        overlap_length = overlap_end - overlap_start
        primary_length = primary_end - primary_start

        overlap_proportion = overlap_length / primary_length

        return overlap_proportion > overlap_threshold

    def find_non_colliding_positions(
        self,
        new_object,
        start_position,
        start_orientation,
        primary_obj_bb,
        max_distance=1,
        step_size=0.00005,
    ):
        """Find a position for the object being added.

        The criteria are such that the object does not:
        i) have a physical collision with other objects (i.e. collision meshes
        intersect)
        ii) "collide" with the initial view of the primary target object, i.e. obscure
        the ability of the agent to start on the primary target at the beginning of an
        experiment

        Args:
            new_object: The object being added
            start_position: The starting position of the new object
            start_orientation: The initial orientation of the new object
            primary_obj_bb: Bounding box of the primary target object (list of two
                defining corners)
            max_distance: The maximum distance to attempt moving the new object
            step_size: The step size for moving the new object

        Returns:
            The newly added object (position updated)

        Raises:
            RuntimeError: If failed to find a non-colliding position
        """
        direction = self.non_conflicting_vector()
        direction /= np.linalg.norm(direction)

        # Move the second object along the direction vector until they no longer collide
        for distance in np.arange(0, max_distance, step_size):
            obj_pos = start_position + distance * direction
            new_object.translation = obj_pos

            # Extract updated bounding box of new object being added
            min_corner, max_corner = get_bounding_corners(new_object)

            # Step the physics simulation to allow objects to settle and compute
            # collisions
            self._sim.step_physics(0.0001)  # 0.0001 appears to be the smallest possible
            # time step

            physical_collision = self._sim.contact_test(new_object.object_id)
            if physical_collision:
                # Reset the pose of the object if any collision, as the physics
                # timestep can cause the object to rotate from this
                new_object.rotation = sim_utils.quat_to_magnum(start_orientation)

            viewpoint_collision = self.check_viewpoint_collision(
                primary_obj_bb=primary_obj_bb, new_obj_bb=[min_corner, max_corner]
            )

            if not physical_collision and not viewpoint_collision:
                # No collision, so not necessary to reset the pose
                return new_object

        raise RuntimeError("Failed to find non-colliding positions")

    @property
    def num_objects(self):
        """Return the number of instantiated objects."""
        rigid_mgr = self._sim.get_rigid_object_manager()
        return rigid_mgr.get_num_objects()

    @property
    def action_space(self):
        """Returns a set with all available actions.

        All available actions are those registered by all agents in the
        environment.
        """
        return set(self._action_space)

    def get_agent(self, agent_id):
        """Return habitat agent instance."""
        agent_index = self._agent_id_to_index[agent_id]
        return self._sim.get_agent(agent_index)

    def apply_actions(self, actions: Sequence[Action]) -> Dict[str, Dict]:
        """Execute given actions in the environment.

        Args:
            actions: The actions to execute

        Returns:
            A dictionary with the observations grouped by agent_id

        Raises:
            TypeError: If the action type is invalid
            ValueError: If the action name is invalid
        """
        if not actions:
            return self.observations

        for action in actions:
            action_name = self.action_name(action)
            if action_name not in self._action_space:
                raise ValueError(f"Invalid action name: {action_name}")

            # TODO: This is for the purpose of type checking, but would be better
            #       handled using the action space check above, once those are
            #       integrated into the type system.
            if not isinstance(
                action,
                (
                    LookDown,
                    LookUp,
                    MoveForward,
                    MoveTangentially,
                    OrientHorizontal,
                    OrientVertical,
                    SetAgentPitch,
                    SetAgentPose,
                    SetSensorPitch,
                    SetSensorPose,
                    SetSensorRotation,
                    SetYaw,
                    TurnLeft,
                    TurnRight,
                ),
            ):
                raise TypeError(f"Invalid action type: {type(action)}")

            action.act(self)

            observations = self.observations

        return observations

    @property
    def observations(self) -> dict:
        """Get sensor observations.

        Returns:
            A dictionary with all sensor observations grouped by sensor module.
                For example:
                    {
                        "agent1": {
                            "sensor1": {
                                "rgba": [....],
                                "depth": [....],
                            :
                        },
                        "agent2": {
                            "sensor2":
                                "rgba": [....],
                                "depth": [....],
                            :
                        }
                    }
        """
        agent_indices = range(len(self._agents))
        obs = self._sim.get_sensor_observations(agent_ids=agent_indices)
        obs = self.process_observations(obs)
        return obs

    def process_observations(self, obs) -> dict:
        """Habitat returns observations grouped by agent_index.

        Initially, we group observations by agent_id instead and call all agents
        to further process the observations.

        Args:
            obs: The observations to process

        Returns:
            The processed observations grouped by agent_id.
        """
        processed_obs = defaultdict(dict)
        for agent_index, agent_obs in obs.items():
            agent = self._agents[agent_index]
            agent_id = self._agents[agent_index].agent_id
            processed_obs[agent_id] = agent.process_observations(agent_obs)

        return processed_obs

    @property
    def states(self) -> dict:
        """Get agent and sensor states (position, rotation, etc..).

        Returns:
            A dictionary with the agent pose in world coordinates and any other
            agent specific state as well as every sensor pose relative to the agent
            as well as any sensor specific state that is not returned by
            :attr:`observations`.

            For example:
                {
                    "camera": {
                        "position": [2.125, 1.5, -5.278],
                        "rotation": [0.707107, 0.0, 0.0.707107, 0.0],
                        "sensors" : {
                            "rgba": {
                                "position": [0.0, 1.5, 0.0],
                                "rotation": [1.0, 0.0, 0.0, 0.0],
                            },
                            "depth": {
                                "position": [0.0, 1.5, 0.0],
                                "rotation": [1.0, 0.0, 0.0, 0.0],
                            },
                            :
                        }
                    },
                    :
                }
        """
        result = {}
        for agent_index, sim_agent in enumerate(self._sim.agents):
            # Get agent and sensor poses from simulator
            agent_node = sim_agent.scene_node

            sensors = {}
            for sensor_id, sensor in agent_node.node_sensors.items():
                rotation = sim_utils.quat_from_magnum(sensor.node.rotation)
                sensors[sensor_id] = {
                    "position": sensor.node.translation,
                    "rotation": rotation,
                }

            # Update agent/module state
            agent_id = self._agents[agent_index].agent_id
            rotation = sim_utils.quat_from_magnum(agent_node.rotation)
            result[agent_id] = {
                "position": agent_node.translation,
                "rotation": rotation,
                "sensors": sensors,
            }

        return result

    def reset(self):
        # All agents managed by this simulator
        agent_indices = range(len(self._agents))
        obs = self._sim.reset(agent_ids=agent_indices)
        obs = self.process_observations(obs)
        return obs

    def close(self) -> None:
        """Close simulator and release resources."""
        sim = getattr(self, "_sim", None)
        if sim is not None:
            sim.close()
            self._sim = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
