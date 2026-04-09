# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from functools import partial
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from mujoco import mjtGeom
from unittest_parametrize import ParametrizedTestCase, parametrize

from tbp.monty.frameworks.actions.actions import LookUp
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR
from tbp.monty.simulators.mujoco import MissingObjectModel, UnknownObjectType
from tbp.monty.simulators.mujoco.simulator import (
    DEFAULT_RESOLUTION,
    PRIMITIVE_OBJECTS,
    MissingObjectTexture,
    MuJoCoSimulator,
)

# Parameters for add primitive object tests
SHAPE_PARAMS = [(s, v) for s, v in PRIMITIVE_OBJECTS.items()]

AGENT_ID = AgentID("agent_id_0")

CUSTOM_OBJECT_DATA_PATH = (
    Path(__file__).parents[2] / "resources" / "test_custom_objects"
)


class MuJoCoSimulatorTestCase(ParametrizedTestCase):
    """Tests for the MuJoCo simulator."""

    def test_initial_scene_is_empty(self) -> None:
        with MuJoCoSimulator() as sim:
            assert sim.model.ngeom == 0
            assert len(sim.spec.geoms) == 0

    @parametrize(
        "shape,shape_type",
        SHAPE_PARAMS,
    )
    def test_add_primitive_object(self, shape: str, shape_type: mjtGeom) -> None:
        with MuJoCoSimulator() as sim:
            sim.add_object(shape)

            assert sim.model.ngeom == 1
            assert len(sim.spec.geoms) == 1
            # 1. Check that the spec was updated
            geom = sim.spec.geom(f"{shape}_0")
            assert geom.type == shape_type
            assert geom.name == f"{shape}_0"

            # 2. Check that the model was updated
            # This raises if it doesn't exist
            sim.model.geom(f"{shape}_0")

    def test_multiple_objects_have_different_ids(self) -> None:
        """Test that multiple objects have different IDs.

        To prevent name collisions in the MuJoCo spec, the names of objects are
        suffixed with their "object number", an increasing index of the objects in the
        scene. So, several objects should be numbered in the order they were added.
        """
        shapes = ["box", "box", "box"]
        with MuJoCoSimulator() as sim:
            for shape in shapes:
                sim.add_object(shape)

            count = len(shapes)
            assert sim.model.ngeom == count
            assert len(sim.spec.geoms) == count
            for i, shape in enumerate(shapes):
                geom = sim.spec.geom(f"{shape}_{i}")
                assert geom is not None
                # Raises if geom doesn't exist with ID
                sim.model.geom(f"{shape}_{i}")

    def test_remove_all_objects(self) -> None:
        with MuJoCoSimulator() as sim:
            sim.add_object("box")
            sim.add_object("capsule")

            assert sim.model.ngeom == 2
            assert len(sim.spec.geoms) == 2
            assert sim.spec.geom("box_0") is not None
            assert sim.spec.geom("capsule_1") is not None

            sim.remove_all_objects()
            assert sim.model.ngeom == 0
            assert len(sim.spec.geoms) == 0
            assert sim.spec.geom("box_0") is None
            assert sim.spec.geom("capsule_1") is None

    def test_primitive_object_positioning(self) -> None:
        with MuJoCoSimulator() as sim:
            sim.add_object("box", position=(1.0, 1.0, 2.0))

            geom = sim.spec.geom("box_0")
            assert geom.type == mjtGeom.mjGEOM_BOX
            assert np.allclose(geom.pos, np.array([1.0, 1.0, 2.0]))

    def test_primitive_box_scaling(self) -> None:
        with MuJoCoSimulator() as sim:
            sim.add_object("box", scale=(3.0, 3.0, 3.0))

            geom = sim.spec.geom("box_0")
            assert geom.type == mjtGeom.mjGEOM_BOX
            assert np.allclose(geom.size, np.array([3.0, 3.0, 3.0]))

    def test_primitive_sphere_scaling(self) -> None:
        """Test that scaling works correctly on a sphere."""
        with MuJoCoSimulator() as sim:
            sim.add_object("sphere", scale=(3.0, 3.0, 3.0))

            assert np.allclose(
                sim.model.geom("sphere_0").size, np.array([3.0, 3.0, 3.0])
            )

    def test_custom_object_loading(self) -> None:
        """Test custom object loading."""
        with MuJoCoSimulator(data_path=CUSTOM_OBJECT_DATA_PATH) as sim:
            sim.add_object("valid_object")

            geom = sim.spec.geom("valid_object_0")
            assert geom.type == mjtGeom.mjGEOM_MESH

            mesh = sim.spec.mesh(geom.meshname)
            assert np.allclose(mesh.refpos, [1.0, 1.0, 1.0])
            assert np.allclose(
                mesh.refquat, [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0, 0.0]
            )

    def test_custom_object_missing(self) -> None:
        with MuJoCoSimulator(data_path=CUSTOM_OBJECT_DATA_PATH) as sim, pytest.raises(
            UnknownObjectType
        ):
            sim.add_object("invalid_object")

    def test_custom_object_missing_texture(self) -> None:
        with MuJoCoSimulator(data_path=CUSTOM_OBJECT_DATA_PATH) as sim, pytest.raises(
            MissingObjectTexture
        ):
            sim.add_object("missing_texture")

    def test_custom_object_missing_model(self) -> None:
        with MuJoCoSimulator(data_path=CUSTOM_OBJECT_DATA_PATH) as sim, pytest.raises(
            MissingObjectModel
        ):
            sim.add_object("missing_model")

    def test_custom_object_missing_metadata(self) -> None:
        with MuJoCoSimulator(data_path=CUSTOM_OBJECT_DATA_PATH) as sim:
            sim.add_object("missing_metadata")

            geom = sim.spec.geom("missing_metadata_0")
            assert geom.type == mjtGeom.mjGEOM_MESH

            mesh = sim.spec.mesh(geom.meshname)
            assert np.allclose(mesh.refpos, ZERO_VECTOR)
            assert np.allclose(mesh.refquat, IDENTITY_QUATERNION)

    def test_agent_that_does_not_understand_an_action(self) -> None:
        """Ensure the simulator works with an agent that doesn't respond to actions."""
        agent_mock = Mock(id=AGENT_ID)
        agent_mock.max_sensor_resolution = DEFAULT_RESOLUTION
        AgentMockClass = Mock(return_value=agent_mock)  # noqa: N806
        sim = MuJoCoSimulator(
            agents=[partial(AgentMockClass)],
            data_path=None,
        )

        with sim:
            action = LookUp(AGENT_ID, rotation_degrees=5.0)
            sim.step([action])

    def test_agent_action_with_attribute_error(self) -> None:
        """This test ensures that the simulator doesn't swallow agent errors."""

        def actuate_look_up(*args, **kwargs):  # noqa: ARG001
            # Simulate an attribute error as from a programming mistake
            raise AssertionError("AgentMock does not have attribute 'foo'")

        agent_mock = Mock(id=AGENT_ID)
        agent_mock.actuate_look_up = Mock(side_effect=actuate_look_up)
        agent_mock.max_sensor_resolution = DEFAULT_RESOLUTION
        AgentMockClass = Mock(return_value=agent_mock)  # noqa: N806
        sim = MuJoCoSimulator(
            agents=[partial(AgentMockClass)],
            data_path=None,
        )
        action = LookUp(AGENT_ID, rotation_degrees=5.0)

        with pytest.raises(AssertionError), sim:
            sim.step([action])
