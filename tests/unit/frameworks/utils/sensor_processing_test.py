# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
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


@st.composite
def orthonormal_vectors(draw):
    v, n = draw(nonzero_orthogonal_vectors())
    return normalize(v), n


@st.composite
def curvature_values(draw):
    k1 = draw(st.floats(min_value=-1e3, max_value=1e3))
    k2 = draw(st.floats(min_value=-1e3, max_value=1e3))
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
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        ks=curvature_values(),
        vectors=orthonormal_vectors(),
    )
    @example(
        angle=0.0,
        ks=(4.0, 2.0),
        vectors=(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
    )  # aligned with pc1_dir -> k1
    @example(
        angle=np.pi / 2,
        ks=(4.0, 2.0),
        vectors=(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
    )  # perpendicular to pc1_dir -> k2
    @example(angle=np.pi / 4, ks=(4.0, 2.0), vectors=(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ))  # 45 degrees -> average
    @example(angle=0.0, ks = (-3.0, -1.0), vectors =(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
    ))  # negative curvatures
    def test_euler_formula(self, angle, ks, vectors):
        """Result matches k1*cos^2(theta) + k2*sin^2(theta) for any direction."""
        pc1, pc2 = vectors
        k1, k2 = ks
        direction = pc1 * np.cos(angle) + pc2 * np.sin(angle)
        result = directional_curvature(
            direction, k1=k1, k2=k2, pc1_dir=pc1, pc2_dir=pc2
        )
        expected = k1 * np.cos(angle) ** 2 + k2 * np.sin(angle) ** 2
        self.assertAlmostEqual(result, expected)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        vectors=orthonormal_vectors(),
    )
    def test_equal_curvatures_returns_that_value(self, angle, k, vectors):
        """When k1 == k2, result equals that value for any in-plane direction."""
        pc1, pc2 = vectors
        # Create a vector in the same plane as pc1 and pc2.
        direction = pc1 * np.cos(angle) + pc2 * np.sin(angle)
        result = directional_curvature(
            direction,
            k1=k,
            k2=k,
            pc1_dir=pc1,
            pc2_dir=pc2,
        )
        self.assertAlmostEqual(result, k)

    @given(
        vectors=orthonormal_vectors(),
        ks=curvature_values(),
        a_scaler=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False).filter(
            lambda x: x != 0.0
        ),
    )
    def test_non_orthogonal_dirs_raises(self, vectors, ks, a_scaler):
        pc1, _ = vectors
        k1, k2 = ks
        bad_pc2 = pc1 * a_scaler
        with self.assertRaises(ValueError):
            directional_curvature(
                movement_direction=np.array([1.0, 0.0, 0.0]),
                k1=k1,
                k2=k2,
                pc1_dir=pc1,
                pc2_dir=bad_pc2,
            )

    @given(vectors=orthonormal_vectors(), ks=curvature_values())
    def test_out_of_plane_direction_raises(self, vectors, ks):
        pc1, pc2 = vectors
        k1, k2 = ks
        movement_direction = np.cross(pc1, pc2)
        with pytest.raises(ValueError):
            directional_curvature(
                movement_direction=movement_direction,
                k1=k1,
                k2=k2,
                pc1_dir=pc1,
                pc2_dir=pc2,
            )

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        vectors=orthonormal_vectors(),
        ks=curvature_values(),
    )
    def test_opposite_direction_same_result(self, angle, vectors, ks):
        """Negating the direction does not change the curvature (sign-invariant)."""
        pc1, pc2 = vectors
        k1, k2 = ks
        direction = pc1 * np.cos(angle) + pc2 * np.sin(angle)
        fwd = directional_curvature(direction, k1=k1, k2=k2, pc1_dir=pc1, pc2_dir=pc2)
        bwd = directional_curvature(-direction, k1=k1, k2=k2, pc1_dir=pc1, pc2_dir=pc2)
        self.assertAlmostEqual(fwd, bwd)
