# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.sensor_processing import (
    directional_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
)
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    nonzero_orthogonal_vectors,
)

# Curvature is reciprocal of the radius, thus 1e3 corresponds
# to 1 mm radius (sharp edge)
MIN_K = -1e3
MAX_K = 1e3


@st.composite
def orthonormal_vectors(draw):
    v, n = draw(nonzero_orthogonal_vectors())
    return normalize(v), n


@st.composite
def curvature_values(draw):
    k1 = draw(st.floats(min_value=MIN_K, max_value=MAX_K))
    k2 = draw(st.floats(min_value=MIN_K, max_value=MAX_K))
    assume(k1 >= k2)
    return k1, k2


class DirectionalCurvatureTest(unittest.TestCase):
    @given(vectors=orthonormal_vectors(), ks=curvature_values())
    def test_zero_direction_returns_zero(self, vectors, ks):
        pc1, pc2 = vectors
        k1, k2 = ks
        result = directional_curvature(
            np.array([0.0, 0.0, 0.0]),
            k1=k1,
            k2=k2,
            pc1_dir=pc1,
            pc2_dir=pc2,
        )
        npt.assert_allclose(result, 0.0, atol=DEFAULT_TOLERANCE)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi),
        ks=curvature_values(),
        vectors=orthonormal_vectors(),
    )
    def test_euler_formula(self, angle, ks, vectors):
        pc1, pc2 = vectors
        k1, k2 = ks
        # Create a vector in the same plane as pc1 and pc2.
        direction = pc1 * np.cos(angle) + pc2 * np.sin(angle)
        result = directional_curvature(
            direction, k1=k1, k2=k2, pc1_dir=pc1, pc2_dir=pc2
        )
        expected = k1 * np.cos(angle) ** 2 + k2 * np.sin(angle) ** 2
        tol = max(
            DEFAULT_TOLERANCE * abs(k1),
            DEFAULT_TOLERANCE * abs(k2),
            DEFAULT_TOLERANCE,
        )
        npt.assert_allclose(result, expected, atol=tol, rtol=DEFAULT_TOLERANCE)

    @given(
        vectors=orthonormal_vectors(),
        a_scaler=st.floats(min_value=-1e3, max_value=1e3).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        ),
    )
    def test_non_orthogonal_pcs_raises(self, vectors, a_scaler):
        pc1, _ = vectors
        bad_pc2 = pc1 * a_scaler
        expected_msg = r"The pc1_dir and pc2_dir must be orthogonal\."
        with pytest.raises(ValueError, match=expected_msg):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=bad_pc2,
            )

    @given(vectors=orthonormal_vectors())
    def test_out_of_plane_movement_raises(self, vectors):
        pc1, pc2 = vectors
        movement_direction = np.cross(pc1, pc2)
        expected_msg = (
            r"The movement_direction must lie in the plane"
            r" of pc1_dir and pc2_dir\."
        )
        with pytest.raises(ValueError, match=expected_msg):
            directional_curvature(
                movement_direction=movement_direction,
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=pc2,
            )

    @given(vectors=orthonormal_vectors())
    def test_pcs_not_unit_vectors_raises(self, vectors):
        pc1, pc2 = vectors
        scaled_pc1 = pc1 * 2.0
        with pytest.raises(ValueError, match="must be unit vectors"):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=scaled_pc1,
                pc2_dir=pc2,
            )

        scaled_pc2 = pc2 * 2.0
        with pytest.raises(ValueError, match="must be unit vectors"):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=scaled_pc2,
            )
