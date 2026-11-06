# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest

import numpy as np
import numpy.typing as npt
from hypothesis import given

from tbp.monty.frameworks.utils.graph_matching_utils import (
    custom_distances,
    euclidean_distances,
)
from tests.unit.frameworks.utils.sensor_processing_test import curvatures
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    unit_vectors,
    vectors_3d,
)


class EuclideanDistancesTest(unittest.TestCase):
    def test_two_hyps_with_two_nn_neighbors_calibration(self) -> None:
        predicted_locations = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        nearest_node_locations = np.array(
            [[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]]
        )
        distances = euclidean_distances(
            predicted_locations=predicted_locations,
            nearest_node_locations=nearest_node_locations,
        )
        # norm([0, 0, 0] - [3, 4, 0]) = 5
        # norm([0, 0, 0] - [0, 0, 2]) = 2
        # norm([1, 1, 1] - [1, 1, 1]) = 0
        # norm([1, 1, 1] - [1, 1, 0]) = 1
        np.testing.assert_allclose(distances, [[5.0, 2.0], [0.0, 1.0]])


class CustomDistancesTest(unittest.TestCase):
    def test_one_hyp_with_two_nn_neighbors_calibration(self) -> None:
        predicted_locations = np.array([[0.0, 0.0, 0.0]])
        surface_normal = np.array([[1.0, 0.0, 0.0]])
        curvature = 1.5
        # First node in tangent plane, second node along surface normal
        nearest_node_locations = np.array([[[0.0, 3.0, 4.0], [1.0, 0.0, 0.0]]])

        distances = custom_distances(
            predicted_locations=predicted_locations,
            nearest_node_locations=nearest_node_locations,
            surface_normals=surface_normal,
            curvature=curvature,
        )

        # For first node, dot product to SN is 0
        # For second node, dot product to SN is 1, and
        # dist = 1 + 1 * (1 / (1.5 + 0.5)) = 1 + 1/2 = 1.5
        np.testing.assert_allclose(distances, [[5.0, 1.5]])

    @given(
        predicted_location=vectors_3d(),
        nearest_node_location=vectors_3d(),
        surface_normal=unit_vectors,
        curvature=curvatures,
    )
    def test_greater_than_or_equal_to_euclidean_dist(
        self,
        predicted_location: npt.NDArray[np.float64],
        nearest_node_location: npt.NDArray[np.float64],
        surface_normal: npt.NDArray[np.float64],
        curvature: float,
    ) -> None:
        predicted_locations = predicted_location.reshape(1, 3)
        nearest_node_locations = nearest_node_location.reshape(1, 1, 3)
        surface_normals = surface_normal.reshape(1, 3)

        euclidean_dists = euclidean_distances(
            predicted_locations=predicted_locations,
            nearest_node_locations=nearest_node_locations,
        )
        custom_dists = custom_distances(
            predicted_locations=predicted_locations,
            nearest_node_locations=nearest_node_locations,
            surface_normals=surface_normals,
            curvature=curvature,
        )

        self.assertTrue(np.all(custom_dists >= euclidean_dists))
