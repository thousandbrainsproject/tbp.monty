# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as ScipyRotation

from tbp.monty.math import DEFAULT_TOLERANCE


def to_scalar_last(wxyz: npt.ArrayLike) -> np.ndarray:
    # TODO(scottcanoe): Remove when no longer handling scalar-last quaternions.
    return np.asarray(wxyz)[..., [1, 2, 3, 0]]


def to_scalar_first(xyzw: npt.ArrayLike) -> np.ndarray:
    # TODO(scottcanoe): Remove when no longer handling scalar-last quaternions.
    return np.asarray(xyzw)[..., [3, 0, 1, 2]]


def scipy_rotations_approx_equal(
    a: ScipyRotation,
    b: ScipyRotation,
    atol: float = DEFAULT_TOLERANCE,
    degrees: bool = False,
) -> bool | np.ndarray:
    """Backport of `scipy.spatial.transform.Rotation.approx_equal` from later versions.

    Args:
        a: First scipy rotation.
        b: Second scipy rotation.
        atol: Absolute tolerance.
        degrees: Whether atol is in degrees. Default is False.

    Returns:
        True if the angular delta between `a` and `b` is below threshold. False
        otherwise. If `a` and `b` are non-single, returns an array of booleans.
    """
    if degrees:
        atol = np.degrees(atol)
    return (b * a.inv()).magnitude() <= atol


class Rotation:
    """Rotation in 3 dimensions.

    This class was created as a replacement for `scipy.spatial.transform.Rotation`
    that better conforms to our conventions. Primarily, we wanted to be consistent about
    using scalar-first (wxyz) order for quaternions, but scalar-last (xyzw) is scipy's
    default mode. Consequently, this class's `from_quat` and `as_quat` implementations
    assume scalar-first order.

    Since `Rotation` is mostly just a wrapper for `scipy.spatial.transform.Rotation`,
    it API closely mirrors it's wrapped counterpart (as of scipy version 1.10.1).
    The notable exceptions are
      - `from_quat` and `as_quat` assume scalar-first order.
      - `from_scipy_rotation` and `as_scipy_rotation` methods have been implemented.
      - The `approx_equal` method has been backported.
    """

    _rot: ScipyRotation

    def __init__(self, obj: ScipyRotation | Rotation) -> None:
        if isinstance(obj, ScipyRotation):
            rot = obj
        elif isinstance(obj, Rotation):
            rot = obj.as_scipy_rotation()
        else:
            raise TypeError(f"Invalid object type: {type(obj)}")
        object.__setattr__(self, "_rot", rot)

    @property
    def single(self) -> bool:
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
    def from_mrp(mrp: npt.ArrayLike) -> Rotation:
        return Rotation(ScipyRotation.from_mrp(mrp))

    def as_mrp(self) -> np.ndarray:
        return self._rot.as_mrp()

    @staticmethod
    def from_quat(quat: npt.ArrayLike) -> Rotation:
        """Build from quaternion(s) in **WXYZ** (scalar-first) order.

        This methods differs substantially from
        `scipy.spatial.transform.Rotation.from_quat`. Here, we expect quaternions
        in scalar-first (wxyz) order, where SciPy expects them in scalar-last (xyzw)
        order. Scalar-last ordering will not be supported.

        Args:
            quat: Array-like of shape (4,) or (N, 4) in scalar-first (wxyz) order.

        Returns:
            A `Rotation` instance.

        Raises:
            ValueError: If the wxyz is not of shape (4,) or (N, 4).
        """
        quat = np.asarray(quat)  # wrap this in try/except?
        if quat.ndim not in {1, 2} or quat.shape[-1] != 4:
            raise ValueError(
                f"Quaternion must be of shape (4,) or (N, 4), got {quat.shape}"
            )
        return Rotation(ScipyRotation.from_quat(quat[..., [1, 2, 3, 0]]))

    def as_quat(self) -> np.ndarray:
        """Quaternion as **WXYZ** (scalar-first). SciPy uses XYZW internally.

        This methods differs substantially from
        `scipy.spatial.transform.Rotation.as_quat`. Here, we return quaternions
        in scalar-first (wxyz) order, whereas scipy returns them in scalar-last (xyzw)
        order. Scalar-last ordering will not be supported.

        Returns:
            Array of shape ``(4,)`` or ``(N, 4)`` in WXYZ order.
        """
        return self._rot.as_quat()[..., [3, 0, 1, 2]]

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
    def concatenate(rotations: Sequence[Rotation]) -> Rotation:
        scipy_rots = [obj.as_scipy_rotation() for obj in rotations]
        return Rotation(ScipyRotation.concatenate(scipy_rots))

    @staticmethod
    def align_vectors(
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        return_sensitivity: bool = False,
    ) -> tuple[Rotation, float] | tuple[Rotation, float, np.ndarray]:
        result = ScipyRotation.align_vectors(a, b, weights, return_sensitivity)
        return (Rotation(result[0]), *result[1:])

    def inv(self) -> Rotation:
        return Rotation(self._rot.inv())

    def apply(self, vectors: npt.ArrayLike, inverse: bool = False) -> np.ndarray:
        return self._rot.apply(vectors, inverse=inverse)

    def magnitude(self) -> float | np.ndarray:
        return self._rot.magnitude()

    def mean(self, weights: npt.ArrayLike | None = None) -> Rotation:
        return Rotation(self._rot.mean(weights))

    def reduce(
        self,
        left: Rotation | None = None,
        right: Rotation | None = None,
        return_indices: bool = False,
    ) -> Rotation | tuple[Rotation, np.ndarray, np.ndarray]:
        if not return_indices:
            return Rotation(self._rot.reduce(left, right, return_indices=False))
        result = self._rot.reduce(left, right, return_indices=True)
        return (Rotation(result[0]), *result[1:])

    def approx_equal(
        self,
        other: Rotation,
        atol: float = DEFAULT_TOLERANCE,
        degrees: bool = False,
    ) -> bool | np.ndarray:
        return scipy_rotations_approx_equal(
            self._rot, other.as_scipy_rotation(), atol=atol, degrees=degrees
        )

    def __bool__(self) -> bool:
        return bool(self._rot)

    def __getitem__(self, indexer: int | slice | None) -> Rotation:
        return Rotation(self._rot[indexer])

    def __len__(self) -> int:
        return len(self._rot)

    def __mul__(self, other: Rotation) -> Rotation:
        return Rotation(self._rot * other.as_scipy_rotation())

    def __repr__(self) -> str:
        if self.single:
            x, y, z = self.as_euler("xyz", degrees=True)
            return f"Rotation(x: {x:.3f}, y: {y:.3f}, z: {z:.3f} [deg])"
        return f"Rotation(length: {len(self)})"

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Rotation is immutable")

    def __getstate__(self) -> ScipyRotation:
        return self._rot

    def __setstate__(self, state: ScipyRotation) -> None:
        object.__setattr__(self, "_rot", state)


