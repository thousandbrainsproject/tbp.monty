# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import numpy.testing as nptest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import MIN_ATTENTION_WEIGHT
from tbp.monty.frameworks.models.evidence_matching.region_proposal.inhibit_recognized_object import (  # noqa: E501
    InhibitRecognizedObject,
    thicken_surface,
)

MAX_POINTS = 20
MAX_LAYERS = 10
MAX_THICKNESS = 1.0
# Surface coordinates are kept small so the offsets along the normals stay
# well above float resolution.
MAX_COORDINATE = 10.0

point_coordinates = st.floats(
    min_value=-MAX_COORDINATE, max_value=MAX_COORDINATE, allow_nan=False
)
thicknesses = st.floats(min_value=1e-3, max_value=MAX_THICKNESS)
layer_counts = st.integers(min_value=1, max_value=MAX_LAYERS)


def surface_points(num_points: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(dtype=np.float64, shape=(num_points, 3), elements=point_coordinates)


def unit_normals(num_points: int) -> st.SearchStrategy[np.ndarray]:
    # Directions drawn away from the origin, then normalised.
    directions = arrays(
        dtype=np.float64,
        shape=(num_points, 3),
        elements=st.floats(min_value=0.1, max_value=1.0),
    )
    return directions.map(lambda d: d / np.linalg.norm(d, axis=1, keepdims=True))


@st.composite
def surfaces(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    num_points = draw(st.integers(min_value=1, max_value=MAX_POINTS))
    return draw(surface_points(num_points)), draw(unit_normals(num_points))


class ThickenSurfaceTest(unittest.TestCase):
    @given(surface=surfaces(), thickness=thicknesses, num_layers=layer_counts)
    def test_keeps_the_surface_points_first(
        self,
        surface: tuple[np.ndarray, np.ndarray],
        thickness: float,
        num_layers: int,
    ) -> None:
        points, normals = surface
        shell = thicken_surface(points, normals, thickness, num_layers)

        self.assertEqual(shell.shape, (len(points) * (1 + 2 * num_layers), 3))
        nptest.assert_array_equal(shell[: len(points)], points)

    @given(surface=surfaces(), thickness=thicknesses, num_layers=layer_counts)
    def test_offsets_every_point_along_its_normal_by_evenly_spaced_signed_steps(
        self,
        surface: tuple[np.ndarray, np.ndarray],
        thickness: float,
        num_layers: int,
    ) -> None:
        # The documented rule: num_layers offsets per side, spread evenly over
        # (0, thickness], each point displaced along its own normal.
        points, normals = surface
        steps = np.linspace(0.0, thickness, num_layers + 1)[1:]
        expected_offsets = np.concatenate([-steps[::-1], steps])
        shell = thicken_surface(points, normals, thickness, num_layers)

        layers = shell[len(points) :].reshape(len(points), 2 * num_layers, 3)
        displacements = layers - points[:, None, :]
        along_normal = np.einsum("pij,pj->pi", displacements, normals)
        nptest.assert_allclose(
            along_normal,
            np.broadcast_to(expected_offsets, along_normal.shape),
            atol=1e-9,
        )
        across_normal = displacements - along_normal[:, :, None] * normals[:, None, :]
        nptest.assert_allclose(across_normal, 0.0, atol=1e-9)


@dataclass
class FakeRegionContext:
    """A RegionContext whose answers are fixed up front.

    ``to_body_frame`` translates by ``body_offset`` so the test can tell model
    from body coordinates.
    """

    recognized_object: str | None
    current_location: np.ndarray | None
    points: np.ndarray
    normals: np.ndarray
    body_offset: np.ndarray

    @property
    def possible_matches(self) -> list[str]:
        return [] if self.recognized_object is None else [self.recognized_object]

    def surface(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:  # noqa: ARG002
        return self.points, self.normals

    def to_body_frame(self, points: np.ndarray, object_id: str) -> np.ndarray:  # noqa: ARG002
        return points + self.body_offset


@st.composite
def recognized_contexts(draw: st.DrawFn) -> FakeRegionContext:
    points, normals = draw(surfaces())
    return FakeRegionContext(
        recognized_object="mug",
        current_location=draw(surface_points(1))[0],
        points=points,
        normals=normals,
        body_offset=draw(surface_points(1))[0],
    )


class InhibitRecognizedObjectTest(unittest.TestCase):
    @given(context=recognized_contexts())
    def test_proposes_nothing_while_no_object_is_recognized(
        self,
        context: FakeRegionContext,
    ) -> None:
        context.recognized_object = None

        self.assertIsNone(InhibitRecognizedObject()(context))

    @given(context=recognized_contexts())
    def test_proposes_nothing_without_a_current_location(
        self,
        context: FakeRegionContext,
    ) -> None:
        context.current_location = None

        self.assertIsNone(InhibitRecognizedObject()(context))

    @given(
        context=recognized_contexts(), thickness=thicknesses, num_layers=layer_counts
    )
    def test_proposes_the_thickened_surface_in_body_coordinates(
        self,
        context: FakeRegionContext,
        thickness: float,
        num_layers: int,
    ) -> None:
        expected = (
            thicken_surface(context.points, context.normals, thickness, num_layers)
            + context.body_offset
        )
        proposer = InhibitRecognizedObject(thickness=thickness, num_layers=num_layers)

        region = proposer(context)

        nptest.assert_allclose(region.locations, expected)

    @given(
        context=recognized_contexts(), weight=st.floats(min_value=-1.0, max_value=0.0)
    )
    def test_gives_every_point_the_configured_weight(
        self,
        context: FakeRegionContext,
        weight: float,
    ) -> None:
        region = InhibitRecognizedObject(weight=weight)(context)

        nptest.assert_array_equal(region.weights, np.full(len(region), weight))

    def test_inhibits_by_default(self) -> None:
        context = FakeRegionContext(
            recognized_object="mug",
            current_location=np.zeros(3),
            points=np.zeros((1, 3)),
            normals=np.array([[0.0, 0.0, 1.0]]),
            body_offset=np.zeros(3),
        )

        region = InhibitRecognizedObject()(context)

        nptest.assert_array_equal(
            region.weights, np.full(len(region), MIN_ATTENTION_WEIGHT)
        )

    def test_calls_surface_and_to_body_frame_for_the_recognized_object(self) -> None:
        context = MagicMock()
        context.recognized_object = sentinel.object_id
        context.surface.return_value = (np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]))
        context.to_body_frame.return_value = np.zeros((0, 3))
        shell_patch = patch(
            "tbp.monty.frameworks.models.evidence_matching.region_proposal"
            ".inhibit_recognized_object.thicken_surface",
            return_value=sentinel.shell,
        )
        with shell_patch:
            InhibitRecognizedObject()(context)

        context.surface.assert_called_once_with(sentinel.object_id)
        context.to_body_frame.assert_called_once_with(
            sentinel.shell, sentinel.object_id
        )
