# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Sequence

import quaternion as qt
from mujoco import (
    MjData,
    MjModel,
    MjsBody,
    MjSpec,
    Renderer,
    mj_forward,
    mjtGeom,
    mjtTexture,
    mjtTextureRole,
)
from typing_extensions import Self, override

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import Agent, AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import Resolution2D
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco.objects import (
    ObjectMetadata,
    load_object_metadata,
)

logger = logging.getLogger(__name__)


# Map of names to MuJoCo primitive object types
PRIMITIVE_OBJECTS = {
    "box": mjtGeom.mjGEOM_BOX,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
    "sphere": mjtGeom.mjGEOM_SPHERE,
}

DEFAULT_RESOLUTION = Resolution2D((64, 64))


class UnknownObjectType(RuntimeError):
    """An unknown object type is requested."""


class MissingObjectModel(RuntimeError):
    """An object type is missing an object model file."""


class MissingObjectTexture(RuntimeError):
    """An object type is missing a texture file."""


class DataPathNotConfigured(RuntimeError):
    """The simulator data_path is not configured and a custom object is requested."""


class MuJoCoSimulator(SimulatedObjectEnvironment):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    def __init__(
        self,
        agents: Sequence[Callable[[MuJoCoSimulator], Agent]] | None = None,
        data_path: str | Path | None = None,
    ) -> None:
        if not agents:
            agents: Sequence[Callable[[MuJoCoSimulator], Agent]] = []

        self.spec = MjSpec()
        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)
        self.data_path = Path(data_path) if data_path else None

        self._agent_partials = agents
        self._agents: dict[AgentID, Agent] = {}
        self._create_agents()
        self._loaded_custom_types: set[str] = set()

        if self._agents:
            self._render_resolution = self._max_sensor_resolution()
        else:
            # This only really comes up in test contexts, but if we don't have
            # any agents, we can't calculate the maximum sensor resolution, so
            # we need to just use a default value.
            self._render_resolution = DEFAULT_RESOLUTION

        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0

        self.renderer = None
        self._recompile()

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        # The spec might be new, so reset all the options
        self.spec.option.gravity = (0.0, 0.0, 0.0)
        g = self.spec.visual.global_
        g.offwidth, g.offheight = self._render_resolution
        self.model, self.data = self.spec.recompile(self.model, self.data)
        # The renderer has to be recreated when the model is updated.
        self._create_renderer()
        # Step the simulation so all objects are in their initial positions.
        mj_forward(self.model, self.data)

    def _create_renderer(self):
        """Create a new MuJoCo renderer, closing the existing one if needed."""
        if self.renderer:
            self.renderer.close()
        self.renderer = Renderer(
            width=self._render_resolution[0],
            height=self._render_resolution[1],
            model=self.model,
        )

    def _create_agents(self) -> None:
        self._agents = {}
        for agent_partial in self._agent_partials:
            agent = agent_partial(self)
            self._agents[agent.id] = agent

    def _max_sensor_resolution(self) -> Resolution2D:
        """Determine the maximum resolution of all the sensors.

        We need this to set the off-screen buffer size in MuJoCo to support the
        highest resolution sensor configured.

        Returns:
            max_width, max_height
        """
        max_width = max_height = 0
        for agent in self._agents.values():
            width, height = agent.max_sensor_resolution
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return Resolution2D((max_width, max_height))

    def remove_all_objects(self) -> None:
        # TODO: is there a better way to do this?
        self.spec = MjSpec()
        self._create_agents()
        self._recompile()
        self._object_count = 0
        self._loaded_custom_types = set()

    @override
    def add_object(
        self,
        name: str,
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectInfo:
        # TODO: remove this once object init samplers return the correct types
        if isinstance(rotation, qt.quaternion):
            rotation = (
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            )

        obj_name = f"{name}_{self._object_count}"

        if name in PRIMITIVE_OBJECTS:
            self._add_primitive_object(obj_name, name, position, rotation, scale)
        else:
            self._add_custom_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()

        if not semantic_id:
            semantic_id = SemanticID(self._object_count)
        return ObjectInfo(
            object_id=ObjectID(self._object_count),
            semantic_id=semantic_id,
        )

    def _add_custom_object(
        self,
        obj_name: str,
        object_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ):
        """Adds a custom object loaded from the data_path to the scene.

        This assumes that each object's files are stored in a directory in the
        `data_path` matching the shape_type. It should contain the mesh in
        'textured.obj', the texture in 'texture_map.png', as well as a 'metadata.json'
        file with additional information we need to correctly add the object to
        the scene.

        Arguments:
            obj_name: Name for the object in the scene.
            object_type: Type of object to add, determines directory to look in.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Initial scale of the object.
        """
        if object_type not in self._loaded_custom_types:
            self._load_custom_object(object_type)

        self.spec.worldbody.add_geom(
            name=obj_name,
            type=mjtGeom.mjGEOM_MESH,
            meshname=f"{object_type}_mesh",
            material=f"{object_type}_mat",
            size=scale,
            pos=position,
            quat=rotation,
        )
        self._loaded_custom_types.add(object_type)

    def _load_custom_object(self, object_type: str) -> None:
        """Loads a custom object from the data_path into the spec.

        This should only be done once per custom object type.

        Raises:
            DataPathNotConfigured: if data_path is not configured
            UnknownObjectType: When the directory for the object_type is missing.
            MissingObjectTexture: When the texture map is missing.
            MissingObjectModel: When the object is missing.
        """
        if not self.data_path:
            raise DataPathNotConfigured(
                "Cannot load custom objects in simulator, "
                "'data_path' is not configured."
            )
        path = self.data_path / object_type
        texture_path = path / "texture_map.png"
        model_path = path / "textured.obj"

        if not path.exists():
            raise UnknownObjectType(f"Unknown object type: {object_type}")
        if not texture_path.exists():
            raise MissingObjectTexture(
                f"The {object_type} is missing 'texture_map.png'."
            )
        if not model_path.exists():
            raise MissingObjectModel(f"The {object_type} is missing 'textured.obj'.")

        # MuJoCo doesn't seem to be able to load the referenced texture from the
        # 'texture.obj' file directly, so we have to load the texture separately and
        # create a material for it that we can add to the mesh.
        self.spec.add_texture(
            name=f"{object_type}_tex",
            type=mjtTexture.mjTEXTURE_2D,
            file=str(texture_path),
        )
        mat = self.spec.add_material(
            name=f"{object_type}_mat",
        )
        mat.textures[mjtTextureRole.mjTEXROLE_RGB] = f"{object_type}_tex"

        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = load_object_metadata(metadata_path, object_type)
        else:
            metadata = ObjectMetadata()

        self.spec.add_mesh(
            name=f"{object_type}_mesh",
            file=str(model_path),
            refquat=metadata.refquat,
            refpos=metadata.refpos,
        )

    def _add_primitive_object(
        self,
        obj_name: str,
        object_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ) -> None:
        """Adds a built-in MuJoCo primitive geom to the scene spec.

        Arguments:
            obj_name: Name for the object in the scene.
            object_type: The primitive object type to add.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Initial scale of the object.

        Raises:
            UnknownObjectType: When the shape_type is unknown.
        """
        world_body: MjsBody = self.spec.worldbody

        try:
            geom_type = PRIMITIVE_OBJECTS[object_type]
        except KeyError:
            raise UnknownObjectType(
                f"Unknown MuJoCo primitive: {object_type}"
            ) from None

        # TODO: should we encapsulate primitive objects into bodies?
        world_body.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation,
        )

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

    @override
    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        logger.debug(f"{actions=}")
        for action in actions:
            agent = self._agents[action.agent_id]
            try:
                action.act(agent)
            except AttributeError as exc:
                # Only catch missing actuate methods, propagate any other errors
                # TODO: In Python>=3.10, use exc.name to determine the missing method
                #   directly instead of relying on a regex match on the message string.
                match = re.search(r"attribute 'actuate_.*'", str(exc))
                if match:
                    logger.warning(f"{agent} does not understand {action}")
                    continue
                raise
        mj_forward(self.model, self.data)
        return self.observations, self.states

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        for agent in self._agents.values():
            agent.reset()
        mj_forward(self.model, self.data)
        return self.observations, self.states

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
        self.renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
