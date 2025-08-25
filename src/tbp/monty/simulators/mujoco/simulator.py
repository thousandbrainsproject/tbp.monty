# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any, Dict, List, Optional

import mujoco
from mujoco import MjData, MjModel, MjsBody, MjSpec, mjtGeom

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)
from tbp.monty.simulators.mujoco.agents import MuJoCoAgent
from tbp.monty.simulators.simulator import Simulator


class UnknownShapeType(RuntimeError):
    """Raised when an unknown shape is requested."""


class MuJoCoSimulator(Simulator):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    def __init__(self, agents: Optional[List[MuJoCoAgent]] = None) -> None:
        self.spec = MjSpec()
        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0
        # Create a map of agent IDs to agents and initialize them
        agents = agents if agents is not None else []
        self._agents = {a.agent_id: a for a in agents}
        self._initialize_agents()

        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)
        # Take an initial step to position all the bodies in the scene
        mujoco.mj_step(self.model, self.data)

    def _initialize_agents(self) -> None:
        for agent in self._agents.values():
            agent.initialize(self.spec)

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        self.model, self.data = self.spec.recompile(self.model, self.data)

    def initialize_agent(self, agent_id, agent_state) -> None:
        pass

    def remove_all_objects(self) -> None:
        # TODO: come up with a better way to remove only the objects added, not the
        #  agent bodies as well.
        self.spec = MjSpec()
        self._recompile()
        self._object_count = 0
        self._initialize_agents()

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics: bool = False,
        object_to_avoid: bool = False,
        primary_target_bb: Optional[List] = None,
    ) -> None:
        obj_name = f"{name}_{self._object_count}"

        # TODO: support arbitrary objects from a registry
        self._add_primitive_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()
        # TODO: We need to step the simulation in order to position the new object in
        #  the scene. They don't just appear at their position at first. This might
        #  cause us some problems later if we decide to use the physics simulation.
        mujoco.mj_step(self.model, self.data)

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
            UnknownShapeType: when the shape_type is unknown
        """
        world_body: MjsBody = self.spec.worldbody

        # TODO: should we encapsulate primitive objects into bodies?

        if shape_type == "sphere":
            geom_type = mjtGeom.mjGEOM_SPHERE
        elif shape_type == "capsule":
            geom_type = mjtGeom.mjGEOM_CAPSULE
        elif shape_type == "ellipsoid":
            geom_type = mjtGeom.mjGEOM_ELLIPSOID
        elif shape_type == "cylinder":
            geom_type = mjtGeom.mjGEOM_CYLINDER
        elif shape_type == "box":
            geom_type = mjtGeom.mjGEOM_BOX
        else:
            raise UnknownShapeType(f"Unknown MuJoCo primitive: {shape_type}")

        world_body.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation,
        )

    def get_num_objects(self) -> int:
        return self._object_count

    def get_action_space(self) -> None:
        pass

    def get_agent(
        self,
        agent_id: str,  # TODO - replace with newtype
    ) -> None:
        pass

    def get_observations(self) -> dict[Any, Any]:
        observations = {}
        for agent in self._agents.values():
            observations[agent.agent_id] = agent.observe(self.model, self.data)
        return observations

    def get_states(self) -> None:
        pass

    def apply_action(self, action: Action) -> Dict[str, Dict]:
        return {}

    def reset(self) -> None:
        self.spec = MjSpec()
        self._object_count = 0
        self._initialize_agents()
        self.model = self.spec.compile()
        self.data = MjData(self.model)

    def close(self) -> None:
        # MuJoCo doesn't actually have anything to close manually
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
