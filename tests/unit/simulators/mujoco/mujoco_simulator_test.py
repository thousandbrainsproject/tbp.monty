# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import numpy as np
from unittest_parametrize import ParametrizedTestCase, param, parametrize

from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

SHAPES = ["box", "capsule", "cylinder", "ellipsoid", "sphere"]
SHAPE_PARAMS = [param(s, id=s) for s in SHAPES]


class MuJoCoSimulatorTestCase(ParametrizedTestCase):
    """Tests for the MuJoCo simulator."""

    def test_initial_scene_is_empty(self) -> None:
        sim = MuJoCoSimulator()
        assert sim.model.ngeom == 0
        assert sim.get_num_objects() == 0
        assert len(sim.spec.geoms) == 0

    @parametrize(
        "shape",
        SHAPE_PARAMS,
    )
    def test_add_primitive_object(self, shape: str) -> None:
        sim = MuJoCoSimulator()
        sim.add_object(shape)

        assert sim.model.ngeom == 1
        assert sim.get_num_objects() == 1
        assert len(sim.spec.geoms) == 1

        # 1. Check that the spec was updated
        spec_xml = sim.spec.to_xml()
        # Sphere is the default and so its type doesn't end up in the resulting XML
        if shape != "sphere":
            assert f'type="{shape}"' in spec_xml
        assert f'name="{shape}_0"' in spec_xml
        # 2. Check that the model was updated
        # This raises if it doesn't exist
        sim.model.geom(f"{shape}_0")

    def test_multiple_objects_have_different_ids(self) -> None:
        """Test that multiple objects have different IDs.

        To prevent name collisions in the MuJoCo spec, the names of objects are
        suffixed with their "object number", an increasing index of the objects in the
        scene. So, several objects should be numbered in the order they were added.
        """
        shapes = ["box", "capsule", "cylinder"]
        sim = MuJoCoSimulator()
        for shape in shapes:
            sim.add_object(shape)

        assert sim.model.ngeom == len(shapes)
        assert sim.get_num_objects() == len(shapes)
        assert len(sim.spec.geoms) == len(shapes)

        spec_xml = sim.spec.to_xml()
        for i, shape in enumerate(shapes):
            assert f'name="{shape}_{i}"' in spec_xml
            # Raises if geom doesn't exist with ID
            sim.model.geom(f"{shape}_{i}")

    def test_remove_all_objects(self) -> None:
        sim = MuJoCoSimulator()
        sim.add_object("box")
        sim.add_object("capsule")

        assert sim.model.ngeom == 2
        assert sim.get_num_objects() == 2
        assert len(sim.spec.geoms) == 2

        sim.remove_all_objects()

        assert sim.model.ngeom == 0
        assert sim.get_num_objects() == 0
        assert len(sim.spec.geoms) == 0

    def test_primitive_object_positioning(self) -> None:
        sim = MuJoCoSimulator()
        sim.add_object("box", position=(1.0, 1.0, 2.0))

        assert np.allclose(sim.model.geom("box_0").pos, np.array([1.0, 1.0, 2.0]))
        assert f'pos="1 1 2"' in sim.spec.to_xml()
