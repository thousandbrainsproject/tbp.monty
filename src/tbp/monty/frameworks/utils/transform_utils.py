# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import quaternion  # noqa: F401 required by numpy-quaternion package
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def numpy_to_scipy_quat(quat):
    """Convert from wxyz to xyzw format of quaternions.

    i.e. identity rotation in scipy is (0,0,0,1).

    Args:
        quat: A quaternion in wxyz format

    Returns:
        A quaternion in xyzw format
    """
    new_quat = np.array((quat[1], quat[2], quat[3], quat[0]))

    return new_quat


def scipy_to_numpy_quat(quat: np.ndarray) -> np.quaternion:
    numpy_quat = np.quaternion(quat[3], quat[0], quat[1], quat[2])
    return numpy_quat


def rotation_as_quat(rot: Rotation, scalar_first: bool = True) -> np.ndarray:
    """Convert a scipy rotation its quaternion representation.

    Scipy added a `scalar_first` argument to `Rotation.as_quat` in version 1.14.0.
    (https://scipy.github.io/devdocs/release/1.14.0-notes.html). This function
    backports that behavior. Note, however, that scipy defaults to scalar-last format.

    Args:
        rot: The scipy rotation object to convert.
        scalar_first: Whether to return the array in (w, x, y, z) or (x, y, z, w) order.
            Defaults to `True`, i.e., (w, x, y, z) order.

    Returns:
        An array with shape (4,) representing a single quaternion, or
        an array with shape (N, 4) representing N quaternions.

    """
    quat = rot.as_quat()
    if scalar_first:
        return quat[..., [3, 0, 1, 2]]
    return quat


def rotation_from_quat(quat: ArrayLike, scalar_first: bool = True) -> Rotation:
    """Create a scipy rotation object from a quaternion.

    Scipy added a `scalar_first` argument to `Rotation.from_quat` in version 1.14.0.
    (https://scipy.github.io/devdocs/release/1.14.0-notes.html). This function
    backports that behavior. Note, however, that scipy defaults to scalar-last format.

    Args:
        quat: An array with shape (4,) for a single quaternion, or
            an array with shape (N, 4) for N quaternions.
        scalar_first: Whether the scalar component is first or last. Defaults to `True`,
            i.e., (w, x, y, z) order.

    Returns:
        The scipy rotation object.
    """
    quat = np.asarray(quat)
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    return Rotation.from_quat(quat)
