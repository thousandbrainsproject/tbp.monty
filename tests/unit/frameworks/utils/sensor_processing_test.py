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
from hypothesis import example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.sensor_processing import directional_curvature


class DirectionalCurvatureTest(unittest.TestCase):
    """Unit tests for the directional_curvature function."""

    def setUp(self):
        self.pc1_dir = np.array([1.0, 0.0, 0.0])
        self.pc2_dir = np.array([0.0, 1.0, 0.0])

    def test_zero_direction_returns_zero(self):
        result = directional_curvature(
            np.array([0.0, 0.0, 0.0]),
            k1=5.0,
            k2=3.0,
            pc1_dir=self.pc1_dir,
            pc2_dir=self.pc2_dir,
        )
        self.assertEqual(result, 0.0)

    def test_near_zero_direction_returns_zero(self):
        result = directional_curvature(
            np.array([1e-15, 0.0, 0.0]),
            k1=5.0,
            k2=3.0,
            pc1_dir=self.pc1_dir,
            pc2_dir=self.pc2_dir,
        )
        self.assertEqual(result, 0.0)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        k1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        k2=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    @example(angle=0.0, k1=4.0, k2=2.0)  # aligned with pc1_dir -> k1
    @example(angle=np.pi / 2, k1=4.0, k2=2.0)  # perpendicular to pc1_dir -> k2
    @example(angle=np.pi / 4, k1=4.0, k2=2.0)  # 45 degrees -> average
    @example(angle=0.0, k1=-3.0, k2=-1.0)  # negative curvatures
    def test_euler_formula(self, angle, k1, k2):
        """Result matches k1*cos^2(theta) + k2*sin^2(theta) for any direction."""
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        result = directional_curvature(
            direction, k1=k1, k2=k2, pc1_dir=self.pc1_dir, pc2_dir=self.pc2_dir
        )
        expected = k1 * np.cos(angle) ** 2 + k2 * np.sin(angle) ** 2
        self.assertAlmostEqual(result, expected)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False),
        k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    def test_equal_curvatures_returns_that_value(self, angle, k):
        """When k1 == k2, result equals that value for any in-plane direction."""
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        result = directional_curvature(
            direction,
            k1=k,
            k2=k,
            pc1_dir=self.pc1_dir,
            pc2_dir=self.pc2_dir,
        )
        self.assertAlmostEqual(result, k)

    def test_non_orthogonal_dirs_raises(self):
        bad_pc2_dir = np.array([0.5, 0.5, 0.0])
        with self.assertRaises(ValueError):
            directional_curvature(
                np.array([1.0, 0.0, 0.0]),
                k1=4.0,
                k2=2.0,
                pc1_dir=self.pc1_dir,
                pc2_dir=bad_pc2_dir,
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
