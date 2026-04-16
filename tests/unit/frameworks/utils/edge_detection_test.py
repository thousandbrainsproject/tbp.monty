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
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.edge_detection import gradient_to_tangent_angle

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)


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


if __name__ == "__main__":
    unittest.main()
