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

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation as ScipyRotation

from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tbp.monty.geometry import scipy_rotations_approx_equal
from tbp.monty.math import DEFAULT_TOLERANCE


@st.composite
def random_scipy_rotations(draw, num: int | None = None):
    shape: int | tuple[int, ...] = (num, 3) if num is not None else 3
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-180, max_value=180),
        ).map(lambda angles: ScipyRotation.from_euler("xyz", angles, degrees=True))
    )


def to_scalar_last(wxyz: npt.ArrayLike) -> np.ndarray:
    return np.asarray(wxyz, dtype=np.float64)[..., [1, 2, 3, 0]]


def to_scalar_first(xyzw: npt.ArrayLike) -> np.ndarray:
    return np.asarray(xyzw, dtype=np.float64)[..., [3, 0, 1, 2]]


def set_rotation_angle(
    rotation: ScipyRotation,
    angle: float,
    degrees: bool,
) -> ScipyRotation:
    """Change the angle amount w.r.t. the axis-angle representation.

    Note that if the rotation's axis is nearly zero, the returned rotation will
    have the rotation axis (1, 0, 0). This is necessary since the rotation axis
    must be normalized before being rescaled.

    Args:
        rotation: The rotation to set the amount of.
        angle: The amount to set the rotation to.
        degrees: Whether the angle is in degrees.

    Returns:
        A new rotation that rotates about its axis by the specified amount.
    """
    rotvec = rotation.as_rotvec()
    if np.linalg.norm(rotvec) >= DEFAULT_TOLERANCE:
        rotvec = normalize(rotvec)
    else:
        rotvec = np.array([1.0, 0.0, 0.0])
    angle = np.degrees(angle) if degrees else angle
    return ScipyRotation.from_rotvec(rotvec * angle)


class ScipyRotationsApproxEqualTest(unittest.TestCase):
    """Test for the `scipy_rotations_approx_equal` function.

    TODO(scottcanoe): Add tests for non-single rotation objects.
    TODO(scottcanoe): Figure out if there's a way to parameterize tests when
      using hypothesis decorators.
    """

    @given(
        random_scipy_rotations(),
        random_scipy_rotations(),
        st.booleans(),
    )
    def test_result_matches_alternate_implementation(
        self,
        a: ScipyRotation,
        b: ScipyRotation,
        degrees: bool,
    ) -> None:
        # Double-ledger test.
        atol = DEFAULT_TOLERANCE if not degrees else np.degrees(DEFAULT_TOLERANCE)
        expected = (a * b.inv()).magnitude() <= atol
        actual = scipy_rotations_approx_equal(
            a, b, atol=DEFAULT_TOLERANCE, degrees=degrees
        )
        self.assertEqual(actual, expected)

    @given(
        a=random_scipy_rotations(),
        rot=random_scipy_rotations(),
        angle=st.floats(min_value=0, max_value=DEFAULT_TOLERANCE * 0.999),
    )
    def test_returns_true_if_difference_below_tolerance(
        self,
        a: ScipyRotation,
        rot: ScipyRotation,
        angle: float,
    ) -> None:
        # Finer-grained test for near-boundary differences.
        self.assertTrue(
            scipy_rotations_approx_equal(
                a,
                set_rotation_angle(rot, angle, degrees=False) * a,
                atol=DEFAULT_TOLERANCE,
            )
        )

    @given(
        random_scipy_rotations(),
        random_scipy_rotations(),
        st.floats(min_value=DEFAULT_TOLERANCE * 1.001, max_value=DEFAULT_TOLERANCE * 2),
    )
    def test_returns_false_if_difference_above_tolerance(
        self,
        a: ScipyRotation,
        rotate: ScipyRotation,
        angle: float,
    ) -> None:
        # Finer-grained test for near-boundary differences.
        self.assertFalse(
            scipy_rotations_approx_equal(
                a,
                set_rotation_angle(rotate, angle, degrees=False) * a,
                atol=DEFAULT_TOLERANCE,
            )
        )
