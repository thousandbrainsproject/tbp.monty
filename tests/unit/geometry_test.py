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
import numpy.testing as nptest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation as ScipyRotation

from tbp.monty.frameworks.utils.spatial_arithmetics import normalize
from tbp.monty.geometry import (
    Rotation,
    scipy_rotations_approx_equal,
    to_scalar_first,
    to_scalar_last,
)
from tbp.monty.math import DEFAULT_TOLERANCE, ROTATION_TOLERANCE_RADIANS


@st.composite
def scipy_rotations(
    draw,
    num: int | None = None,
):
    shape: int | tuple[int, ...] = (num, 3) if num is not None else 3
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(min_value=-180, max_value=180),
        ).map(lambda angles: ScipyRotation.from_euler("xyz", angles, degrees=True))
    )


@st.composite
def rotation_axes(draw):
    return draw(scipy_rotations().map(lambda rot: as_axis_angle(rot)[0]))


@st.composite
def quaternions(draw, num: int | None = None):
    return draw(
        scipy_rotations(num=num).map(lambda rot: to_scalar_first(rot.as_quat()))
    )


@st.composite
def points_3d(
    draw,
    num: int | None = None,
    min_value: float = -1e6,
    max_value: float = 1e6,
):
    return draw(
        arrays(
            dtype=np.float64,
            shape=(num, 3) if num is not None else 3,
            elements=st.floats(min_value=min_value, max_value=max_value),
        )
    )


def as_axis_angle(
    rot: ScipyRotation,
    epsilon: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, float]:
    """Get the axis-angle representation of a rotation.

    Args:
        rot: The rotation to get the axis-angle representation of.
        epsilon: The epsilon to use for the normalization. Note that this is for
           a normalization threshold and therefore uses DEFAULT_TOLERANCE instead of
           ROTATION_TOLERANCE.

    Returns:
        A tuple of the axis and the angle.
    """
    angle = rot.magnitude()
    rotvec = rot.as_rotvec()
    try:
        axis = normalize(rotvec, epsilon=epsilon)
    except ValueError:
        axis = np.array([1.0, 0.0, 0.0])
    return axis, angle


class TestQuaternionFormatConversions(unittest.TestCase):
    """Validate testing functions that the main tests rely on."""

    def setUp(self):
        w, x, y, z = 0, 1, 2, 3
        self.scalar_first_1d = np.array([w, x, y, z], dtype=np.float64)
        self.scalar_last_1d = np.array([x, y, z, w], dtype=np.float64)
        self.scalar_first_2d = np.stack([self.scalar_first_1d] * 2)
        self.scalar_last_2d = np.stack([self.scalar_last_1d] * 2)

    def test_to_scalar_first_1d(self) -> None:
        nptest.assert_allclose(
            to_scalar_first(self.scalar_last_1d), self.scalar_first_1d
        )

    def test_to_scalar_first_2d(self) -> None:
        nptest.assert_allclose(
            to_scalar_first(self.scalar_last_2d), self.scalar_first_2d
        )

    def test_to_scalar_last_1d(self) -> None:
        nptest.assert_allclose(
            to_scalar_last(self.scalar_first_1d), self.scalar_last_1d
        )

    def test_to_scalar_last_2d(self) -> None:
        nptest.assert_allclose(
            to_scalar_last(self.scalar_first_2d), self.scalar_last_2d
        )


class ScipyRotationsApproxEqualTest(unittest.TestCase):
    """Test for the `scipy_rotations_approx_equal` function."""

    ROTATION_EXAMPLE_ERROR = 1e-15

    @given(
        a=scipy_rotations(),
        axis=rotation_axes(),
        angle=st.floats(
            min_value=0, max_value=ROTATION_TOLERANCE_RADIANS - ROTATION_EXAMPLE_ERROR
        ),
    )
    def test_returns_true_if_delta_below_tolerance(
        self,
        a: ScipyRotation,
        axis: np.ndarray,
        angle: float,
    ) -> None:
        """Finer-grained test of rotation deltas near but below tolerance.

        Note that the rotation amount's upper bound is slightly lower than the
        tested threshold due to floating-point precision. Constructing a rotation
        with a specific rotation amount is accurate up to about 1e-15 radians.
        """
        rot = ScipyRotation.from_rotvec(axis * angle)
        self.assertTrue(scipy_rotations_approx_equal(a, rot * a))

    @given(
        a=scipy_rotations(num=10),
        axis=rotation_axes(),
        angle=st.floats(
            min_value=0, max_value=ROTATION_TOLERANCE_RADIANS - ROTATION_EXAMPLE_ERROR
        ),
    )
    def test_returns_all_true_if_all_deltas_below_tolerance_for_multiple_rotations(
        self,
        a: ScipyRotation,
        axis: np.ndarray,
        angle: float,
    ) -> None:
        """Finer-grained test of rotation deltas near but below tolerance.

        Note that the rotation amount's upper bound is slightly lower than the
        tested threshold due to floating-point precision. Constructing a rotation
        with a specific rotation amount is accurate up to about 1e-15 radians.
        """
        rot = ScipyRotation.from_rotvec(axis * angle)
        self.assertTrue(all(scipy_rotations_approx_equal(a, rot * a)))

    @given(
        a=scipy_rotations(),
        axis=rotation_axes(),
        angle=st.floats(
            min_value=ROTATION_TOLERANCE_RADIANS + ROTATION_EXAMPLE_ERROR,
            max_value=2 * ROTATION_TOLERANCE_RADIANS,
        ),
    )
    def test_returns_false_if_delta_above_tolerance(
        self,
        a: ScipyRotation,
        axis: np.ndarray,
        angle: float,
    ) -> None:
        """Finer-grained test of rotation deltas near but above tolerance.

        Note that the rotation amount's lower bound is slightly higher than the
        tested threshold due to floating-point precision. Constructing a rotation
        with a specific rotation amount is accurate up to about 1e-15 radians.
        """
        rot = ScipyRotation.from_rotvec(axis * angle)
        self.assertFalse(scipy_rotations_approx_equal(a, rot * a))

    @given(
        a=scipy_rotations(num=10),
        axis=rotation_axes(),
        angle=st.floats(
            min_value=ROTATION_TOLERANCE_RADIANS + ROTATION_EXAMPLE_ERROR,
            max_value=2 * ROTATION_TOLERANCE_RADIANS,
        ),
    )
    def test_returns_all_false_if_all_deltas_above_tolerance_for_multiple_rotations(
        self,
        a: ScipyRotation,
        axis: np.ndarray,
        angle: float,
    ) -> None:
        """Finer-grained test of rotation deltas near but above tolerance.

        Note that the rotation amount's lower bound is slightly higher than the
        tested threshold due to floating-point precision. Constructing a rotation
        with a specific rotation amount is accurate up to about 1e-15 radians.
        """
        rot = ScipyRotation.from_rotvec(axis * angle)
        self.assertFalse(any(scipy_rotations_approx_equal(a, rot * a)))

    @given(
        a=scipy_rotations(),
        b=scipy_rotations(),
    )
    def test_against_alternate_implementation_over_full_range(
        self,
        a: ScipyRotation,
        b: ScipyRotation,
    ) -> None:
        """Double-ledger test."""
        expected = (a * b.inv()).magnitude() <= ROTATION_TOLERANCE_RADIANS
        actual = scipy_rotations_approx_equal(a, b, tol=ROTATION_TOLERANCE_RADIANS)
        self.assertEqual(actual, expected)

    @given(
        a=scipy_rotations(),
        axis=rotation_axes(),
        angle=st.floats(
            min_value=0.9 * ROTATION_TOLERANCE_RADIANS,
            max_value=1.1 * ROTATION_TOLERANCE_RADIANS,
        ),
    )
    def test_against_alternate_implementation_near_boundary(
        self,
        a: ScipyRotation,
        axis: np.ndarray,
        angle: float,
    ) -> None:
        """Double-ledger test focused on and around the tolerance threshold."""
        b = ScipyRotation.from_rotvec(axis * angle) * a
        expected = (a * b.inv()).magnitude() <= ROTATION_TOLERANCE_RADIANS
        actual = scipy_rotations_approx_equal(a, b, tol=ROTATION_TOLERANCE_RADIANS)
        self.assertEqual(actual, expected)


class RotationQuaternionTest(unittest.TestCase):
    @given(
        quat=quaternions(),
        xyz=points_3d(),
    )
    def test_from_quat_assumes_scalar_first_order(
        self,
        quat: np.ndarray,
        xyz: np.ndarray,
    ) -> None:
        rot = Rotation.from_quat(quat)
        scipy_rot = ScipyRotation.from_quat(to_scalar_last(quat))
        nptest.assert_allclose(
            rot.apply(xyz), scipy_rot.apply(xyz), atol=ROTATION_TOLERANCE_RADIANS
        )

    @given(quat=quaternions())
    def test_as_quat_returns_scalar_first_order(self, quat: np.ndarray) -> None:
        rot = Rotation.from_quat(quat)
        scipy_rot = ScipyRotation.from_quat(to_scalar_last(quat))
        nptest.assert_allclose(
            rot.as_quat(),
            to_scalar_first(scipy_rot.as_quat()),
            atol=ROTATION_TOLERANCE_RADIANS,
        )
