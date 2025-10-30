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
    return np.array((quat[1], quat[2], quat[3], quat[0]))


def scipy_to_numpy_quat(quat: np.ndarray) -> np.quaternion:
    return np.quaternion(quat[3], quat[0], quat[1], quat[2])


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


def cartesian_to_spherical(coords: ArrayLike) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Converts Cartesian (x, y, z) coordinates to spherical (radius, azimuth, elevation)
    coordinates under the assumption that
     - +x points right, +y points up, and +z points backward
     - azimuth is measured away from the forward (-z) axis, and elevation is
       measured upward from the horizontal (xz) plane.

    Azimuth is bound to [-pi, pi), and elevation is bound to [-pi/2, pi/2]. Undefined
    angles are returned as 0.

    Args:
        coords: x, y, z coordinates with shape (3,) for a single point or
            (N, 3) for multiple points.

    Returns:
        A (3,) or (N, 3) shaped array of spherical coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    # We want to work with 1D arrays, not scalars.
    if coords.ndim == 1:
        x, y, z = coords.reshape(1, 3).T
    else:
        x, y, z = coords.T

    radius = np.sqrt(x**2 + y**2 + z**2)
    radius_xz = np.sqrt(x**2 + z**2)
    azimuth = -np.arctan2(x, -z)
    elevation = np.arctan2(y, radius_xz)

    # Enforce undefined angles to be 0.
    is_vertical = np.isclose(radius_xz, 0)
    if np.any(is_vertical):
        azimuth = np.where(is_vertical, 0, azimuth)
        elevation = np.where(np.isclose(radius, 0), 0, elevation)

    spherical = np.column_stack([radius, azimuth, elevation])
    return spherical[0] if coords.ndim == 1 else spherical


def spherical_to_cartesian(coords: ArrayLike) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates.

    Converts (radius, azimuth, elevation) coordinates to (x, y, z) coordinates
    under the assumption that
     - +x points right, +y points up, and +z points backward.
     - azimuth = 0 points down the forward axis (i.e., -z), and elevation is measured
       upward from the horizontal xz-plane.

    Args:
        coords: (radius, azimuth, elevation coordinates) in with shape (3,) for
            a single point or (N, 3) for multiple points.

    Returns:
        A (3,) or (N, 3) shaped array of Cartesian coordinates.
    """
    coords = np.asarray(coords, dtype=float)
    radius, azimuth, elevation = coords if coords.ndim == 1 else coords.T

    y = radius * np.sin(elevation)
    radius_xz = radius * np.cos(elevation)
    x = -radius_xz * np.sin(azimuth)
    z = -radius_xz * np.cos(azimuth)

    return np.array([x, y, z]) if coords.ndim == 1 else np.column_stack([x, y, z])
