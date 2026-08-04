# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import strategies as st

from tbp.monty.geometry import Rotation
from tbp.monty.math import QuaternionWXYZ, VectorXYZ


@st.composite
def position(draw) -> VectorXYZ:
    x = draw(st.floats(min_value=-10.0, max_value=10.0))
    y = draw(st.floats(min_value=-10.0, max_value=10.0))
    z = draw(st.floats(min_value=-10.0, max_value=10.0))
    return x, y, z


@st.composite
def unit_quaternion(draw) -> QuaternionWXYZ:
    """Strategy to generate unit quaternions.

    Returns:
        A unit quaternion as a 4-tuple, scalar first
    """
    # We're generating the quaternions from Euler angles because generating the
    # coefficients directly results in zero-quaterions, which are invalid and raise
    # an error when trying to construct the Rotation.
    x_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    y_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    z_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    rotation = Rotation.from_euler("xyz", [x_rad, y_rad, z_rad], degrees=False)
    normalized_rotation = rotation.as_quat()
    return cast("QuaternionWXYZ", tuple(normalized_rotation))


@st.composite
def x_rotation_quaterion(draw) -> QuaternionWXYZ:
    x_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    rotation = Rotation.from_euler("xyz", [x_rad, 0.0, 0.0], degrees=False)
    normalized_rotation = rotation.as_quat()
    return cast("QuaternionWXYZ", tuple(normalized_rotation))


@st.composite
def constrained_angle(draw) -> tuple[float, float]:
    """Strategy to generate an angle constrained to a constraint.

    Returns:
        Tuple of angle and constraint
    """
    constraint = draw(st.floats(min_value=0.0, max_value=180.0))
    angle = draw(st.floats(min_value=-constraint, max_value=constraint))
    return angle, constraint
