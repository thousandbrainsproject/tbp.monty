# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Dict, List, Optional

import numpy as np
from mujoco import MjData, MjModel, MjsBody, MjSpec, mjtGeom

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)
from tbp.monty.simulators.simulator import Simulator


class MuJoCoSimulator(Simulator):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    def __init__(self) -> None:
        self.spec = MjSpec()
        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)

        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        self.model, self.data = self.spec.recompile(self.model, self.data)

    def initialize_agent(self, agent_id, agent_state) -> None:
        """Update agent runtime state."""
        pass

    def remove_all_objects(self) -> None:
        """Remove all objects from the simulated environment."""
        self.spec = MjSpec()
        self._recompile()
        self._object_count = 0
        # TODO - reinitialize agents since they will have been removed

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics=False,
        object_to_avoid=False,
        primary_target_bb: Optional[List] = None,
    ) -> None:
        """Add new object to simulated environment.

        Adds a new object based on the named object. This assumes that the set of
        available objects are preloaded and keyed by name.

        Args:
            name: Registered object name
            position: Initial absolute position of the object
            rotation: Initial orientation of the object
            scale: Initial object scale
            semantic_id: Optional override object semantic ID
            enable_physics: Whether to enable physics on the object
            object_to_avoid: If True, ensure the object is not colliding with
              other objects
            primary_target_bb: If not None, this is a list of the min and
              max corners of a bounding box for the primary object, used to prevent
              obscuring the primary objet with the new object.
        """
        obj_name = f"{name}_{self._object_count}"

        # TODO: support arbitrary objects from a registry
        self._add_primitive_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()

        # TODO: reinitialize agents?

    def _add_primitive_object(
        self,
        obj_name: str,
        shape_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ) -> None:
        """Adds a builtin MuJoCo primitive geom to the scene spec.

        Arguments:
            obj_name: Identifier for the object in the scene, must be unique.
            shape_type: The primitive shape to add.
            position: Initial position of the object
            rotation: Initial orientation of the object
            scale: Initial scale of the object

        Raises:
            ValueError: when the shape_type is unknown
        """
        world_body: MjsBody = self.spec.worldbody

        # TODO: should we encapsulate primitive objects into bodies?

        if shape_type == "sphere":
            world_body.add_geom(
                name=obj_name,
                type=mjtGeom.mjGEOM_SPHERE,
                # Use first scale component as radius
                size=[scale[0] * 0.5, 0.0, 0.0],
                pos=position,
                quat=rotation,
            )
        elif shape_type == "capsule":
            world_body.add_geom(
                name=obj_name,
                type=mjtGeom.mjGEOM_CAPSULE,
                # Size is radius and half-height from X and Y
                size=[scale[0] * 0.5, scale[1] * 0.5, 0.0],
                pos=position,
                quat=rotation,
            )
        elif shape_type == "ellipsoid":
            world_body.add_geom(
                name=obj_name,
                type=mjtGeom.mjGEOM_ELLIPSOID,
                # Size is radius in all three axes of the local frame
                size=np.array(scale) * 0.5,
                pos=position,
                quat=rotation,
            )
        elif shape_type == "cylinder":
            world_body.add_geom(
                name=obj_name,
                type=mjtGeom.mjGEOM_CYLINDER,
                # Size is radius and half-height, oriented along the Z axis
                # TODO: should this be using X and Z instead of X and Y?
                size=[scale[0] * 0.5, scale[1] * 0.5, 0.0],
                pos=position,
                quat=rotation,
            )
        elif shape_type == "box":
            world_body.add_geom(
                name=obj_name,
                type=mjtGeom.mjGEOM_BOX,
                # MuJoCo box dimensions are half the distance
                size=np.array(scale) * 0.5,
                pos=position,
                quat=rotation,
            )
        else:
            raise ValueError(f"Unknown MuJoCo primitive: {shape_type}")

    def get_num_objects(self) -> int:
        """Return the number of instantiated objects in the environment."""
        return self._object_count

    def get_action_space(self):
        """Returns the set of all available actions."""
        pass

    def get_agent(self, agent_id):
        """Return agent instance."""
        pass

    def get_observations(self):
        """Get sensor observations."""
        pass

    def get_states(self):
        """Get agent and sensor states."""
        pass

    def apply_action(self, action: Action) -> Dict[str, Dict]:
        """Execute the given action in the environment.

        Args:
            action (Action): the action to execute

        Returns:
            (Dict[str, Dict]): A dictionary with the observations grouped by agent_id
        """
        pass

    def reset(self) -> None:
        """Reset the simulator."""
        pass

    def close(self) -> None:
        """Close any resources used by the simulator."""
        pass
