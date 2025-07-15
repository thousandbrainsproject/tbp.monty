# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import xml.etree.ElementTree as ET
from typing import List
from xml.etree.ElementTree import Element

import numpy as np
import pytest
from mujoco import MjSpec
from unittest_parametrize import ParametrizedTestCase, param, parametrize

from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

# Parameters for add primitive object tests
SHAPES = ["box", "capsule", "cylinder", "ellipsoid", "sphere"]
SHAPE_PARAMS = [param(s, id=s) for s in SHAPES]


class MuJoCoSimulatorTestCase(ParametrizedTestCase):
    """Tests for the MuJoCo simulator."""

    def test_initial_scene_is_empty(self) -> None:
        sim = MuJoCoSimulator()
        self.assert_counts_equal(sim, 0)

    @parametrize(
        "shape",
        SHAPE_PARAMS,
    )
    def test_add_primitive_object(self, shape: str) -> None:
        sim = MuJoCoSimulator()
        sim.add_object(shape)

        self.assert_counts_equal(sim, 1)
        # 1. Check that the spec was updated
        geom_elems = self.parse_spec_geoms(sim.spec)
        if shape != "sphere":
            # Sphere is the default and so its type doesn't end up in the resulting XML
            assert geom_elems[0].attrib["type"] == f"{shape}"
        assert geom_elems[0].attrib["name"] == f"{shape}_0"

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
        sim = MuJoCoSimulator()
        for shape in shapes:
            sim.add_object(shape)

        self.assert_counts_equal(sim, len(shapes))
        geom_elems = self.parse_spec_geoms(sim.spec)
        for i, shape in enumerate(shapes):
            # Objects should appear in the spec XML in the same order as they were added
            assert geom_elems[i].attrib["name"] == f"{shape}_{i}"
            # Raises if geom doesn't exist with ID
            sim.model.geom(f"{shape}_{i}")

    def test_remove_all_objects(self) -> None:
        sim = MuJoCoSimulator()
        sim.add_object("box")
        sim.add_object("capsule")

        self.assert_counts_equal(sim, 2)
        sim.remove_all_objects()
        self.assert_counts_equal(sim, 0)

    def test_primitive_object_positioning(self) -> None:
        sim = MuJoCoSimulator()
        sim.add_object("box", position=(1.0, 1.0, 2.0))

        assert np.allclose(sim.model.geom("box_0").pos, np.array([1.0, 1.0, 2.0]))
        geom_elems = self.parse_spec_geoms(sim.spec)
        assert geom_elems[0].attrib["pos"] == "1 1 2"

    def test_primitive_box_scaling(self) -> None:
        sim = MuJoCoSimulator()
        sim.add_object("box", scale=(3.0, 3.0, 3.0))

        assert np.allclose(sim.model.geom("box_0").size, np.array([3.0, 3.0, 3.0]))
        geom_elems = self.parse_spec_geoms(sim.spec)
        assert geom_elems[0].attrib["size"] == "3 3 3"

    def test_primitive_sphere_scaling(self) -> None:
        """Test that scaling works correctly on a sphere."""
        sim = MuJoCoSimulator()
        sim.add_object("sphere", scale=(3.0, 3.0, 3.0))

        assert np.allclose(sim.model.geom("sphere_0").size, np.array([3.0, 3.0, 3.0]))
        geom_elems = self.parse_spec_geoms(sim.spec)
        assert geom_elems[0].attrib["size"] == "3"

    @staticmethod
    def assert_counts_equal(sim: MuJoCoSimulator, count: int) -> None:
        assert sim.model.ngeom == count
        assert sim.get_num_objects() == count
        assert len(sim.spec.geoms) == count

    @staticmethod
    def parse_spec_geoms(spec: MjSpec) -> List[Element]:
        spec_xml = spec.to_xml()
        parsed_xml = ET.fromstring(spec_xml)  # noqa: S314
        world_body = parsed_xml.find("worldbody")
        if world_body is None:
            pytest.fail("Couldn't find <worldbody> in MuJoCo spec")
        return world_body.findall("geom")
