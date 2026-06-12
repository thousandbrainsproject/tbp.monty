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
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence, cast

from mujoco import (
    MjData,
    MjModel,
    MjsBody,
    MjSpec,
    Renderer,
    mj_forward,
    mjtGeom,
    mjtLightType,
    mjtTexture,
    mjtTextureRole,
)
from typing_extensions import Self, override

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.geometry import Rotation
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco.agents import Agent
from tbp.monty.simulators.mujoco.objects import (
    ObjectMetadata,
    load_object_metadata,
)

# Scaling factor to make MuJoCo primitives roughly the same size
# as their Habitat counterparts. This was determined by trial and error.
HABITAT_SCALING_FACTOR = (0.05, 0.05, 0.05)

if TYPE_CHECKING:
    from functools import partial

logger = logging.getLogger(__name__)

# Map of names to MuJoCo primitive object types
PRIMITIVE_OBJECTS = {
    "box": mjtGeom.mjGEOM_BOX,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
    "sphere": mjtGeom.mjGEOM_SPHERE,
}

# Define primitives with the same names as Habitat uses so we don't
# have to define new environment interface configurations during the
# transition period.
# TODO: remove once Habitat is gone and the test configs are updated to use
#   MuJoCo names for these objects.
HABITAT_PRIMITIVE_OBJECTS = {
    "capsule3DSolid": mjtGeom.mjGEOM_CAPSULE,
    "cubeSolid": mjtGeom.mjGEOM_BOX,
}

# Default rendering resolution in the event that there are no sensor
# configurations, e.g. in tests.
DEFAULT_RESOLUTION = Resolution2D(width=64, height=64)


MuJoCoAgentFactory = Callable[["MuJoCoSimulator"], Agent]


class UnknownObjectType(RuntimeError):
    """An unknown object type is requested."""


class MissingObjectModel(RuntimeError):
    """An object type is missing an object model file."""


class MissingObjectTexture(RuntimeError):
    """An object type is missing a texture file."""


class DataPathNotConfigured(RuntimeError):
    """The simulator data_path is not configured and a custom object is requested."""


class ActuateMethodMissing(RuntimeError):
    """The simulator applied an action to an agent that lacks that actuate method."""

    agent_class: type
    method_name: str

    def __init__(self, agent_class: type, method_name: str) -> None:
        msg = f"{agent_class.__name__} does not understand '{method_name}'"
        super().__init__(msg)
        self.agent_class = agent_class
        self.method_name = method_name


class MuJoCoSimulator(SimulatedObjectEnvironment):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    spec: MjSpec
    model: MjModel
    data: MjData

    _data_path: Path | None
    _raise_actuate_missing: bool
    _agent_partials: Sequence[MuJoCoAgentFactory]
    _agents: dict[AgentID, Agent]
    _loaded_custom_types: set[str]
    _object_count: int
    _renderers: dict[tuple[int, int], Renderer]

    def __init__(
        self,
        agents: Sequence[MuJoCoAgentFactory] | None = None,
        data_path: str | Path | None = None,
        raise_actuate_missing: bool = True,
    ) -> None:
        """Constructs a MuJoCo simulated environment.

        Args:
            agents: the agents to set up in the environment.
              These are provided by Hydra as partially applied constructors that
              are missing the `simulator` argument.
            data_path: the path to where custom object data should be loaded from.
            raise_actuate_missing: whether to raise an exception when an agent
              does not have an actuate method for an Action.
        """
        self.spec = MjSpec()
        self.model = self.spec.compile()
        self.data = MjData(self.model)
        self._data_path = Path(data_path) if data_path else None
        self._raise_actuate_missing = raise_actuate_missing

        self._agent_partials = [] if agents is None else agents
        self._agents = {}
        self._create_agents()
        self._loaded_custom_types: set[str] = set()

        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0

        self._renderers = {}
        self._recompile()

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        # The spec might be new, so reset all the options
        self._configure_spec_settings()
        self._configure_lights()
        self.model, self.data = self.spec.recompile(self.model, self.data)
        # The renderers have to be recreated when the model is updated.
        self._close_renderers()
        # Step the simulation so all objects are in their initial positions.
        mj_forward(self.model, self.data)

    def _configure_spec_settings(self) -> None:
        """Set all the relevant global settings on the spec object."""
        self.spec.option.gravity = (0.0, 0.0, 0.0)
        # Configure the maximum rendering resolution for the off-screen buffer.
        # Start with a default resolution in case we don't have agents and therefore
        # sensors to query, e.g. in tests.
        render_resolution = DEFAULT_RESOLUTION
        if self._agents:
            render_resolution = self._max_sensor_resolution()
        g = self.spec.visual.global_
        g.offheight = render_resolution.height
        g.offwidth = render_resolution.width

    def _configure_lights(self) -> None:
        """Configure the lights as needed.

        We're attempting to recreate the lighting setup we were getting from Habitat,
        i.e. a directional light from the front of the object, with ambient lighting
        on the back sides.

        Using a fixed ambient light on the back side doesn't provide good results, but
        putting it on the headlight does.

        Note: this makes a lot of assumptions about the layout of our scene and
          the objects in the scene.
        """
        # TODO: Consider making these configurable.

        # Configure the headlight to produce the ambient lighting we want on
        # the back of the objects.
        self.spec.visual.headlight.ambient = (0.5, 0.5, 0.5)
        self.spec.visual.headlight.diffuse = (0.0, 0.0, 0.0)
        self.spec.visual.headlight.specular = (0.0, 0.0, 0.0)
        # Add a directional light on the "front" side of the object.
        self.spec.worldbody.add_light(
            pos=(0, 0, 0.2),
            diffuse=(0.6, 0.6, 0.6),
            type=mjtLightType.mjLIGHT_DIRECTIONAL,
        )

    def renderer_for_res(self, resolution: Resolution2D) -> Renderer:
        """Creates or returns a renderer of the specified resolution.

        Used by Agents to get sensor specific renderers, since MuJoCo camera
        "resolution" doesn't affect the resolution of the captured images.

        Returns:
            a renderer of the specified resolution
        """
        resolution_key = (resolution.height, resolution.width)
        if resolution_key not in self._renderers:
            self._renderers[resolution_key] = Renderer(
                height=resolution.height, width=resolution.width, model=self.model
            )
        return self._renderers[resolution_key]

    def _create_agents(self) -> None:
        self._agents = {}
        for agent_partial in self._agent_partials:
            agent = agent_partial(self)
            self._agents[agent.id] = agent

    def _max_sensor_resolution(self) -> Resolution2D:
        """Returns the maximum width and heights of the sensors.

        Used by the simulator to determine the size of the off-screen rendering
        surface to ensure it is always large enough for any sensor images we
        need to render.

        Note: the maximum width and maximum height may come from separate sensors.

        Returns:
            max_width, max_height
        """
        max_width = max_height = 0
        # Introspect the agent partials to determine what the original sensor
        # configs were, so we can determine the maximum resolution needed.
        agent_sensor_cfgs: list[dict[SensorID, SensorConfig]] = [
            p.keywords["sensor_configs"]
            for p in cast("list[partial[Agent]]", self._agent_partials)
        ]
        for sensor_cfgs in agent_sensor_cfgs:
            for sensor_cfg in sensor_cfgs.values():
                max_height = max(max_height, sensor_cfg.resolution.height)
                max_width = max(max_width, sensor_cfg.resolution.width)
        return Resolution2D(height=max_height, width=max_width)

    def remove_all_objects(self) -> None:
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
        if semantic_id is not None:
            logger.warning(
                "MuJoCo does not support adding objects with custom semantic IDs."
            )

        obj_name = f"{name}_{self._object_count}"

        if name in PRIMITIVE_OBJECTS:
            self._add_primitive_object(obj_name, name, position, rotation, scale)
        elif name in HABITAT_PRIMITIVE_OBJECTS:
            # Habitat primitive objects are much smaller than the default MuJoCo ones
            # so we need to adjust the default scale.
            scale = (
                scale[0] * HABITAT_SCALING_FACTOR[0],
                scale[1] * HABITAT_SCALING_FACTOR[1],
                scale[2] * HABITAT_SCALING_FACTOR[2],
            )
            self._add_primitive_object(obj_name, name, position, rotation, scale)
        else:
            self._add_custom_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()

        # Using the object count for the semantic_id will give a distinct
        # value for each added object, and _might_ map to MuJoCo's internal
        # object IDs if we need to use those.
        return ObjectInfo(
            object_id=ObjectID(self._object_count),
            semantic_id=SemanticID(self._object_count),
        )

    def _add_custom_object(
        self,
        obj_name: str,
        object_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ) -> None:
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
        if scale != (1.0, 1.0, 1.0):
            # TODO: In order to support this, we need to update the
            #  object loading code to set the scale on the "mesh" object,
            #  which also means we need to track loaded objects with the
            #  scale included.
            raise NotImplementedError(
                "Custom objects do not currently support "
                "'scale' other than (1.0, 1.0, 1.0)."
            )

        if object_type not in self._loaded_custom_types:
            self._load_custom_object(object_type)

        self.spec.worldbody.add_geom(
            name=obj_name,
            type=mjtGeom.mjGEOM_MESH,
            meshname=f"{object_type}_mesh",
            material=f"{object_type}_mat",
            pos=position,
            quat=rotation,
        )

    def _load_custom_object(self, object_type: str) -> None:
        """Loads a custom object from the data_path into the spec.

        This should only be done once per custom object type.

        Raises:
            DataPathNotConfigured: if data_path is not configured
            UnknownObjectType: When the directory for the object_type is missing.
            MissingObjectTexture: When the texture map is missing.
            MissingObjectModel: When the object is missing.
        """
        if not self._data_path:
            raise DataPathNotConfigured(
                "Cannot load custom objects in simulator, "
                "'data_path' is not configured."
            )
        path = self._data_path / object_type
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

        self._loaded_custom_types.add(object_type)

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
        """
        # In Habitat, the capsule is initially oriented with the round portions at the
        # top and bottom along the vertical Y axis. MuJoCo places the capsule with the
        # round ends pointing towards and away from the agent along the Z axis.
        # To mimic Habitat's behaviour, we need to rotate the primitive object to
        # include that orientation change.
        habitat_correction_rot = Rotation.from_euler("x", -90, degrees=True)
        rotation_rot = Rotation.from_quat(rotation)
        rotation_rot = rotation_rot * habitat_correction_rot
        rotation_quat = rotation_rot.as_quat()

        world_body: MjsBody = self.spec.worldbody

        # TODO: remove try and except clauses when Habitat is removed
        try:
            geom_type = PRIMITIVE_OBJECTS[object_type]
        except KeyError:
            geom_type = HABITAT_PRIMITIVE_OBJECTS[object_type]

        # TODO: should we encapsulate primitive objects into bodies?
        world_body.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation_quat,
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
        for action in actions:
            agent = self._agents[action.agent_id]
            logger.debug(f"Applying {action} to {agent}")
            try:
                action.act(agent)  # type: ignore[attr-defined]
            except AttributeError as exc:
                # Only catch missing actuate methods, propagate any other errors
                if exc.name and exc.name.startswith("actuate_"):
                    if self._raise_actuate_missing:
                        raise ActuateMethodMissing(
                            agent_class=exc.obj.__class__, method_name=exc.name
                        ) from None
                    msg = f"{exc.obj} does not understand {exc.name}"
                    logger.warning(msg)
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
        self._close_renderers()

    def _close_renderers(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
