# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as ScipyRotation

from tbp.monty.math import ROTATION_TOLERANCE_RADIANS


def to_scalar_last(wxyz: npt.ArrayLike) -> np.ndarray:
    """Convert a quaternion from scalar-first (wxyz) to scalar-last (xyzw) order.

    This is a helper function for the `Rotation` class extracted for testing purposes.

    Args:
        wxyz: Array-like of shape (4,) or (N, 4) in scalar-first (wxyz) order.

    Returns:
        Array of shape (4,) or (N, 4) in scalar-last (xyzw) order.
    """
    return np.asarray(wxyz)[..., [1, 2, 3, 0]]


def to_scalar_first(xyzw: npt.ArrayLike) -> np.ndarray:
    """Convert a quaternion from scalar-last (xyzw) to scalar-first (wxyz) order.

    This is a helper function for the `Rotation` class extracted for testing purposes.

    Args:
        xyzw: Array-like of shape (4,) or (N, 4) in scalar-last (xyzw) order.

    Returns:
        Array of shape (4,) or (N, 4) in scalar-first (wxyz) order.

    """
    return np.asarray(xyzw)[..., [3, 0, 1, 2]]


def scipy_rotations_approx_equal(
    a: ScipyRotation,
    b: ScipyRotation,
    tol: float = ROTATION_TOLERANCE_RADIANS,
) -> bool | np.ndarray:
    """Backport of `scipy.spatial.transform.Rotation.approx_equal`.

    Args:
        a: First scipy rotation.
        b: Second scipy rotation.
        tol: Absolute tolerance, expressed in radians.

    Returns:
        True if the angular delta between `a` and `b` is within tolerance. False
        otherwise. If `a` and `b` are non-single, returns an array of booleans.
    """
    return (b * a.inv()).magnitude() <= tol


class Rotation:
    """Rotation in 3 dimensions.

    This class was created to be a (nearly) drop-in replacement for
    ()`scipy.spatial.transform.Rotation`)[https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.spatial.transform.Rotation.html].
    that better conforms to our conventions. Primarily, we wanted to be consistent about
    using scalar-first (wxyz) order for quaternions, but scalar-last (xyzw) is scipy's
    default mode. Consequently, this class's `from_quat` and `as_quat` implementations
    assume and return scalar-first quaternion components.

    Since `Rotation` is a thin wrapper around `scipy.spatial.transform.Rotation`,
    its API is largely inherited from scipy. The main exceptions are:
      - `from_quat` and `as_quat` assume scalar-first order.
      - The `approx_equal` method has been backported from future scipy versions.
      - For consistency, `from_scipy_rotation` and `as_scipy_rotation` methods have
        been added.

    Any missing scipy methods can be added as needed.

    """

    _rot: ScipyRotation

    def __init__(self, scipy_rotation: ScipyRotation) -> None:
        self._rot = scipy_rotation

    @property
    def single(self) -> bool:
        """Whether this instance represents a single rotation."""
        return self._rot.single

    @staticmethod
    def from_euler(
        seq: str,
        angles: float | npt.ArrayLike,
        degrees: bool = False,
    ) -> Rotation:
        return Rotation(ScipyRotation.from_euler(seq, angles, degrees=degrees))

    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        return self._rot.as_euler(seq, degrees=degrees)

    @staticmethod
    def from_matrix(matrix: npt.ArrayLike) -> Rotation:
        return Rotation(ScipyRotation.from_matrix(matrix))

    def as_matrix(self) -> np.ndarray:
        return self._rot.as_matrix()

    @staticmethod
    def from_quat(quat: npt.ArrayLike) -> Rotation:
        """Build from (scalar-first) quaternion data.

        This methods differs substantially from
        `scipy.spatial.transform.Rotation.from_quat`. Here, we expect quaternions
        in scalar-first (wxyz) order, where SciPy expects them in scalar-last (xyzw)
        order. Scalar-last ordering will not be supported.

        Args:
            quat: Array-like of shape (4,) or (N, 4) in scalar-first (wxyz) order.

        Returns:
            A `Rotation` instance.

        Raises:
            ValueError: If `quat` does not have shape (4,) or (N, 4).
        """
        try:
            # If the conversion to scalar-last fails because it has the wrong shape,
            # we want to raise the same type of error scipy would.
            quat = np.asarray(quat)
            xyzw = to_scalar_last(quat)
        except IndexError as e:
            raise ValueError(
                f"Expected `quat` to have shape (4,) or (N, 4), got {np.shape(quat)}"
            ) from e
        return Rotation(ScipyRotation.from_quat(xyzw))

    def as_quat(self) -> np.ndarray:
        """The (scalar-first) quaternion representation.

        This methods differs substantially from
        `scipy.spatial.transform.Rotation.as_quat`. Here, we return quaternions
        in scalar-first (wxyz) order, whereas scipy returns them in scalar-last (xyzw)
        order. Scalar-last ordering will not be supported.

        Returns:
            Array of shape (4,) or (N, 4) in scalar-first (wxyz) order.
        """
        return to_scalar_first(self._rot.as_quat())

    @staticmethod
    def from_rotvec(rotvec: npt.ArrayLike, degrees: bool = False) -> Rotation:
        return Rotation(ScipyRotation.from_rotvec(rotvec, degrees=degrees))

    def as_rotvec(self, degrees: bool = False) -> np.ndarray:
        return self._rot.as_rotvec(degrees=degrees)

    @staticmethod
    def from_scipy_rotation(rot: ScipyRotation) -> Rotation:
        return Rotation(rot)

    def as_scipy_rotation(self) -> ScipyRotation:
        return self._rot

    @staticmethod
    def identity(num: int | np.integer | None = None) -> Rotation:
        return Rotation(ScipyRotation.identity(num))

    @staticmethod
    def random(
        num: int | np.integer | None = None,
        random_state: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> Rotation:
        return Rotation(ScipyRotation.random(num, random_state))

    @staticmethod
    def concatenate(rotations: Iterable[Rotation]) -> Rotation:
        """Concatenate a sequence of `Rotation` objects into a single object.

        This is useful if you want to, for example, take the mean of a set of
        rotations and need to pack them into a single object to do so.

        Args:
            rotations: The rotations to concatenate. If a single `Rotation` object is
              passed in, a copy is returned.

        Returns:
            The `Rotation` instance containing the concatenated rotations.
        """
        scipy_rots = [obj.as_scipy_rotation() for obj in rotations]
        return Rotation(ScipyRotation.concatenate(scipy_rots))

    @staticmethod
    def align_vectors(
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        return_sensitivity: bool = False,
    ) -> tuple[Rotation, float] | tuple[Rotation, float, np.ndarray]:
        """Estimate a rotation to optimally align two sets of vectors.

        For full details, see
        (`scipy.spatial.transform.Rotation.align_vectors`)[https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html].

        Args:
            a: Array-like of shape (3,) or (N, 3).
            b: Array-like of shape (3,) or (N, 3).
            weights: Weights describing the relative importance of the vector
              observations. If None (default), then all values in weights are assumed
              to be 1. One and only one weight may be infinity, and weights must be
              positive.
            return_sensitivity: Whether to return the sensitivity matrix. See Notes for
              details. Default is False.

        Returns:
            rotation : Best estimate of the `Rotation` that transforms `b` to `a`.
            rssd : Square root of the weighted sum of the squared distances between
              the given sets of vectors.
            sensitivity_matrix : Sensitivity matrix of the estimated rotation estimate.
              See scipy documentation (link above) for details.
        """
        result = ScipyRotation.align_vectors(a, b, weights, return_sensitivity)
        return (Rotation(result[0]), *result[1:])

    def inv(self) -> Rotation:
        """Create a new `Rotation` that is the inverse of this `Rotation`.

        Composition of a rotation with its inverse is an identity transformation.

        Returns:
            The new `Rotation` object.

        Examples:
            >>> fwd = Rotation.from_euler("y", -np.pi/6)  # yaw right 30°
            >>> inv = fwd.inv()
            >>> inv.inv().approx_equal(fwd)
            True
        """
        return Rotation(self._rot.inv())

    def apply(self, vectors: npt.ArrayLike, inverse: bool = False) -> np.ndarray:
        """Apply this rotation to a set of vectors.

        If the original frame rotates to the final frame by this rotation,
        then its application to a vector can be seen in two ways:
          - As a projection of vector components
            expressed in the final frame
            to the original frame.
          - As the physical rotation of a vector
            being glued to the original frame as it rotates.
            In this case the vector components
            are expressed in the original frame
            before and after the rotation.

        Args:
            vectors: Array-like of shape (3,) or (N, 3) of xyz coordinates to rotate.
            inverse: If `True` then apply the inverse of this rotation.
                Equivalent to `rotation.inv().apply(vectors)`.

        Returns:
            An array of shape (3,) or (N, 3) of rotated xyz coordinates.

        Examples:
            >>> # pitch up 180°, then roll counter-clockwise 90°
            >>> a_rotation = Rotation.from_euler("XZ", [np.pi, np.pi/2])
            >>> a_rotation.apply([3, 5, 8])
            array([-5., -3., -8.])

            >>> vectors = np.array([
            ...     [3, 5, 8],
            ...     [3, 5, -8],
            ...     [3, -5, -8],
            ...     [-3, -5, -8],
            ...     [-3, -5, 8],
            ...     [-3, 5, 8]
            ... ], dtype=float)
            >>> a_rotation.apply(vectors)
            array([[-5., -3., -8.],
                   [-5., -3.,  8.],
                   [ 5., -3.,  8.],
                   [ 5.,  3.,  8.],
                   [ 5.,  3., -8.],
                   [-5.,  3., -8.]])
        """
        return self._rot.apply(vectors, inverse=inverse)

    def magnitude(self) -> float | np.ndarray:
        return self._rot.magnitude()

    def mean(self, weights: npt.ArrayLike | None = None) -> Rotation:
        return Rotation(self._rot.mean(weights))

    def approx_equal(
        self,
        other: Rotation,
        tol: float = ROTATION_TOLERANCE_RADIANS,
    ) -> bool | np.ndarray:
        """Check if this rotation is approximately equal to another rotation.

        Args:
            other: The other rotation to compare to.
            tol: Absolute tolerance, expressed in radians.

        Returns:
            True if the angular delta between `a` and `b` is within tolerance. False
            otherwise. If `a` and `b` are non-single, returns an array of booleans.
        """
        return scipy_rotations_approx_equal(
            self._rot,
            other.as_scipy_rotation(),
            tol=tol,
        )

    def __getitem__(self, indexer: int | slice | None) -> Rotation:
        return Rotation(self._rot[indexer])

    def __len__(self) -> int:
        return len(self._rot)

    def __mul__(self, other: Rotation) -> Rotation:
        """Compose this rotation with the other.

        If `p` and `q` are two rotations, then the composition of 'q` followed
        by `p` is equivalent to `p * q`. In terms of rotation matrices,
        the composition can be expressed as
        `p.as_matrix() @ q.as_matrix()`.

        Args:
            other: The other rotation to compose with.

        Returns:
            The composed rotation.
        """
        return Rotation(self._rot * other.as_scipy_rotation())

    def __repr__(self) -> str:
        quat = self.as_quat()
        if quat.ndim == 1:
            w, x, y, z = quat
        else:
            w = quat[:, 0]
            x = quat[:, 1]
            y = quat[:, 2]
            z = quat[:, 3]
        return f"{self.__class__.__name__}(w={w}, x={x}, y={y}, z={z})"

    def __getstate__(self) -> ScipyRotation:
        return self._rot

    def __setstate__(self, state: ScipyRotation) -> None:
        object.__setattr__(self, "_rot", state)
