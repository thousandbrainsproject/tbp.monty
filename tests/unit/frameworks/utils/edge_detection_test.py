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
from hypothesis import assume, example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.edge_detection import (
    gradient_to_tangent_angle,
    is_geometric_edge,
)

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)

PATCH_SIZE = 64

positive_thresholds = st.floats(min_value=1e-8, max_value=10.0)

STEP_EDGE_IMAGE = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
STEP_EDGE_IMAGE[:, : PATCH_SIZE // 2] = 1.0


@st.composite
def flat_depth_image(draw):
    """Generate a constant-depth patch with a random depth value.

    Returns:
        Float32 array of shape (PATCH_SIZE, PATCH_SIZE) filled with a constant depth.
    """
    depth = draw(st.floats(min_value=0.01, max_value=100.0))
    return np.full((PATCH_SIZE, PATCH_SIZE), depth, dtype=np.float32)


class GradientToTangentAngleTest(unittest.TestCase):
    """Property-based tests for gradient_to_tangent_angle."""

    @given(gradient_angle=angles)
    def test_result_in_range(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        assert 0.0 <= result < 2 * np.pi

    @given(gradient_angle=angles)
    def test_perpendicularity(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        remainder = (result - gradient_angle) % np.pi
        self.assertAlmostEqual(remainder, np.pi / 2)


class IsGeometricEdgeTest(unittest.TestCase):
    """Property-based tests for is_geometric_edge."""

    @given(patch=flat_depth_image(), theta=angles, threshold=positive_thresholds)
    @example(
        patch=np.full((PATCH_SIZE, PATCH_SIZE), 1.0, dtype=np.float32),
        theta=0.0,
        threshold=0.01,
    )
    def test_flat_depth_returns_false(self, patch, theta, threshold):
        self.assertFalse(is_geometric_edge(patch, theta, threshold))

    @given(theta=angles)
    @example(theta=np.pi / 2)
    @example(theta=0.0)
    def test_theta_has_period_pi(self, theta):
        result_base = is_geometric_edge(STEP_EDGE_IMAGE, theta)
        result_shifted = is_geometric_edge(STEP_EDGE_IMAGE, theta + np.pi)
        self.assertEqual(result_base, result_shifted)

    @given(theta=angles, t_low=positive_thresholds, t_high=positive_thresholds)
    @example(theta=np.pi / 2, t_low=1e-6, t_high=1.0)
    def test_lower_threshold_preserves_detection(self, theta, t_low, t_high):
        assume(t_low < t_high)
        if is_geometric_edge(STEP_EDGE_IMAGE, theta, t_high):
            self.assertTrue(is_geometric_edge(STEP_EDGE_IMAGE, theta, t_low))
