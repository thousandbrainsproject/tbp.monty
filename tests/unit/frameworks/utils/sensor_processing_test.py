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
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.sensor_processing import directional_curvature
from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    non_zero_magnitude_vectors,
)


@st.composite
def orthonormal_vectors(draw):
    random_base = normalize(draw(non_zero_magnitude_vectors))
    v = normalize(draw(non_zero_magnitude_vectors))
    n = np.cross(random_base, v)
    assume(np.linalg.norm(n) >= 1e-12)
    return v, n


@st.composite
def curvature_values(draw):
    k1 = draw(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
    k2 = draw(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
    assume(k1 >= k2)
    return k1, k2


class DirectionalCurvatureTest(unittest.TestCase):
    """Unit tests for the directional_curvature function."""

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
        self.assertEqual(result, 0.0)

    @given(vectors=orthonormal_vectors(), ks=curvature_values())
    def test_near_zero_direction_returns_zero(self, vectors, ks):
        pc1, pc2 = vectors
        k1, k2 = ks
        result = directional_curvature(
            np.array([1e-15, 0.0, 0.0]),
            k1=k1,
            k2=k2,
            pc1_dir=pc1,
            pc2_dir=pc2,
        )
        self.assertEqual(result, 0.0)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        ks=curvature_values(),
        vectors=orthonormal_vectors(),
    )
    @example(angle=0.0, k1=4.0, k2=2.0)  # aligned with pc1_dir -> k1
    @example(angle=np.pi / 2, k1=4.0, k2=2.0)  # perpendicular to pc1_dir -> k2
    @example(angle=np.pi / 4, k1=4.0, k2=2.0)  # 45 degrees -> average
    @example(angle=0.0, k1=-3.0, k2=-1.0)  # negative curvatures
    def test_euler_formula(self, angle, ks, vectors):
        """Result matches k1*cos^2(theta) + k2*sin^2(theta) for any direction."""
        pc1, pc2 = vectors
        k1, k2 = ks
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
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
    @settings(max_examples=10000)
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
                np.array([1.0, 0.0, 0.0]),
                k1=k1,
                k2=k2,
                pc1_dir=pc1,
                pc2_dir=bad_pc2,
            )

    def test_non_unit_direction_normalizes(self):
        result = directional_curvature(
            np.array([10.0, 0.0, 0.0]),
            k1=7.0,
            k2=1.0,
            pc1_dir=self.pc1_dir,
            pc2_dir=self.pc2_dir,
        )
        self.assertAlmostEqual(result, 7.0)

    def test_out_of_plane_direction_raises(self):
        with self.assertRaises(ValueError):
            directional_curvature(
                np.array([0.0, 0.0, 1.0]),
                k1=4.0,
                k2=2.0,
                pc1_dir=self.pc1_dir,
                pc2_dir=self.pc2_dir,
            )

    def test_partially_out_of_plane_direction_raises(self):
        with self.assertRaises(ValueError):
            directional_curvature(
                np.array([1.0, 0.0, 1.0]),
                k1=4.0,
                k2=2.0,
                pc1_dir=self.pc1_dir,
                pc2_dir=self.pc2_dir,
            )

    def test_nearly_in_plane_direction_passes(self):
        result = directional_curvature(
            np.array([1.0, 0.0, 1e-8]),
            k1=4.0,
            k2=2.0,
            pc1_dir=self.pc1_dir,
            pc2_dir=self.pc2_dir,
        )
        self.assertAlmostEqual(result, 4.0)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        k1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        k2=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    def test_opposite_direction_same_result(self, angle, k1, k2):
        """Negating the direction does not change the curvature (sign-invariant)."""
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        fwd = directional_curvature(
            direction, k1=k1, k2=k2, pc1_dir=self.pc1_dir, pc2_dir=self.pc2_dir
        )
        bwd = directional_curvature(
            -direction, k1=k1, k2=k2, pc1_dir=self.pc1_dir, pc2_dir=self.pc2_dir
        )
        self.assertAlmostEqual(fwd, bwd)
