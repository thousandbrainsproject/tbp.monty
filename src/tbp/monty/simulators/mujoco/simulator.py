# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mujoco import (
    MjData,
    MjModel,
    MjsBody,
    MjSpec,
    mj_forward,
    mjtGeom,
    mjtTexture,
    mjtTextureRole,
)
from typing_extensions import override

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    ProprioceptiveState,
)
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco import Agent, AgentConfig, Size
from tbp.monty.simulators.simulator import Simulator

if TYPE_CHECKING:
    from os import PathLike


logger = logging.getLogger(__name__)

# Map of names to MuJoCo primitive object types
PRIMITIVE_OBJECT_TYPES = {
    "box": mjtGeom.mjGEOM_BOX,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
    "sphere": mjtGeom.mjGEOM_SPHERE,
}


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

    def __init__(
        self,
        agent_configs: Sequence[AgentConfig],
        data_path: PathLike,
        # TODO: remove after adding remaining arguments
        **kwargs,  # noqa: ARG002
    ) -> None:
        self.spec = MjSpec()
        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)

        self.data_path = Path(data_path)
        self._agent_configs = agent_configs
        self._agents: dict[AgentID, Agent] = {}
        self._create_agents()

        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0

        self._recompile()

    def _max_sensor_resolution(self) -> Size:
        """Determine the maximum resolution of a sensor.

        We need this to set the off-screen buffer size in MuJoCo to support the
        highest resolution sensor configured.

        Returns:
            max_x, max_y
        """
        max_x = 0
        max_y = 0
        for agent_cfg in self._agent_configs:
            for sensor_cfg in agent_cfg["agent_args"]["sensor_configs"].values():
                max_x = max(max_x, sensor_cfg["resolution"][0])
                max_y = max(max_y, sensor_cfg["resolution"][1])
        return max_x, max_y

    def _create_agents(self):
        for agent_config in self._agent_configs:
            agent_type = agent_config["agent_type"]
            agent_args = agent_config["agent_args"]
            agent = agent_type(simulator=self, **agent_args)
            self._agents[agent.id] = agent

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        self.spec.option.gravity = (0.0, 0.0, 0.0)  # TODO: check if necessary.
        g = self.spec.visual.global_
        g.offwidth, g.offheight = self._max_sensor_resolution()
        self.model, self.data = self.spec.recompile(self.model, self.data)
        mj_forward(self.model, self.data)

    def remove_all_objects(self) -> None:
        self.spec = MjSpec()
        self._create_agents()
        self._recompile()
        self._object_count = 0

    @override
    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectInfo:
        obj_name = f"{name}_{self._object_count}"

        if name in PRIMITIVE_OBJECT_TYPES:
            self._add_primitive_object(obj_name, name, position, rotation, scale)
        else:
            self._add_ycb_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()

        return ObjectInfo(
            object_id=ObjectID(self._object_count),
            semantic_id=semantic_id,
        )

    def _add_ycb_object(
        self,
        obj_name: str,
        shape_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ):
        """Adds an object from the YCB dataset.

        This assumes that each object's files are stored in a directory in the
        `data_path` matching their name, and containing 'textured.obj' and
        'texture_map.png'.

        Arguments:
            obj_name: Identifier for the object in the scene, must be unique.
            shape_type: Type of the object in the scene.
            position: Position of the object in the scene.
            rotation: Rotation of the object in the scene.
            scale: Scale of the object in the scene.

        Raises:
            UnknownShapeType: when shape_type is unknown.
        """
        path = self.data_path / shape_type

        if not path.exists():
            raise UnknownShapeType(f"Unknown YCB object: {shape_type}")

        self.spec.add_texture(
            name=f"{shape_type}_tex",
            type=mjtTexture.mjTEXTURE_2D,
            file=f"{path / 'texture_map.png'}",
        )
        mat = self.spec.add_material(
            name=f"{shape_type}_mat",
        )
        mat.textures[mjtTextureRole.mjTEXROLE_RGB] = f"{shape_type}_tex"

        metadata_path = path / "metadata.json"
        metadata = json.load(metadata_path.open())

        self.spec.add_mesh(
            name=f"{shape_type}_mesh",
            file=f"{path / 'textured.obj'}",
            refquat=metadata["refquat"],
            refpos=metadata["refpos"],
        )
        self.spec.worldbody.add_geom(
            name=obj_name,
            type=mjtGeom.mjGEOM_MESH,
            meshname=f"{shape_type}_mesh",
            material=f"{shape_type}_mat",
            size=scale,
            pos=position,
            quat=rotation,
        )

    def _add_primitive_object(
        self,
        obj_name: str,
        shape_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ) -> None:
        """Adds a built-in MuJoCo primitive geom to the scene spec.

        Arguments:
            obj_name: Identifier for the object in the scene, must be unique.
            shape_type: The primitive shape to add.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Initial scale of the object.

        Raises:
            UnknownShapeType: When the shape_type is unknown.
        """
        world_body: MjsBody = self.spec.worldbody

        # TODO: should we encapsulate primitive objects into bodies
        try:
            geom_type = PRIMITIVE_OBJECT_TYPES[shape_type]
        except KeyError:
            raise UnknownShapeType(f"Unknown MuJoCo primitive: {shape_type}") from None

        world_body.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation,
        )

    @override
    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        logger.debug(f"{actions=}")
        for action in actions:
            agent = self._agents[action.agent_id]
            try:
                action.act(agent)
            except AttributeError:
                logger.warning(f"{agent} does not understand {action}")
                continue
        mj_forward(self.model, self.data)
        return self.observations, self.states

    @property
    def observations(self) -> Observations:
        obs = Observations()
        for agent in self._agents.values():
            obs[agent.id] = agent.observations

        return obs

    @property
    def states(self) -> ProprioceptiveState:
        states = ProprioceptiveState()
        for agent in self._agents.values():
            states[agent.id] = agent.state
        return states

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        for agent in self._agents.values():
            agent.reset()
        mj_forward(self.model, self.data)
        return self.observations, self.states

    def close(self) -> None:
        pass
