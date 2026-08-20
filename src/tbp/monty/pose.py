# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR

# FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
# FloatVector = np.ndarray
# FloatVector = npt.NDArray[Any]
FloatVector = npt.NDArray[np.float64]
"""A type alias for a 1D array of `float`"""


def _deg(d: float) -> float:
    """Convert degrees to radians.

    | Radians | Degrees |    sin    |    cos    |
    |---------|---------|-----------|-----------|
    | Pi      |     180 |  0.000000 | -1.000000 |
    | Pi / 2  |      90 |  1.000000 |  0.000000 |
    | Pi / 4  |      45 |  0.707107 |  0.707107 |
    | Pi / 6  |      30 |  0.500000 |  0.866025 |
    | Pi / 12 |      15 |  0.258819 |  0.965926 |
    | Pi / 36 |       5 |  0.087156 |  0.996194 |

    Args:
        d: The angle in degrees

    Returns:
        The angle in radians
    """
    return np.deg2rad(d)  # d * np.pi / 180


def _round_scalar(n: float, ndigits: int = 6) -> float:
    """Round a `float` to `ndigits` places and normalize zeros.

    Returns:
        The rounded `float` value.
    """
    n = round(n, ndigits)
    return 0.0 if n == 0.0 else n


def _round_tuple(t: npt.ArrayLike, ndigits: int = 6) -> tuple[float, ...]:
    """Round an array of `float` to `ndigits` places and normalize zeros.

    Returns:
        A tuple of rounded `float` values.
    """
    t = np.asarray(t, dtype=float)
    return tuple(_round_scalar(n, ndigits) for n in t)


class Location:  # noqa: PLW1641
    r"""A location (position coordinates) in a given _reference frame_.

    Examples:
        >>> a_location = Location()
        >>> a_location
        Location(frame=None, x=0.0, y=0.0, z=0.0)
        >>> print(a_location)
        (0.0, 0.0, 0.0)
        >>> print(a_location.frame)
        None

        >>> a_location = Location(xyz=[8, -3, 5])
        >>> a_location
        Location(frame=None, x=8.0, y=-3.0, z=5.0)
        >>> print(a_location)
        (8.0, -3.0, 5.0)
        >>> a_location.as_array()
        array([ 8., -3.,  5.])

        >>> a_location.x = 3
        >>> a_location.x
        3.0
        >>> a_location.y = 5
        >>> a_location.y
        5.0
        >>> a_location.z = -2
        >>> a_location.z
        -2.0
        >>> a_location.as_array()
        array([ 3.,  5., -2.])

        >>> a_location.move_x(5)
        Location(frame=None, x=8.0, y=5.0, z=-2.0)
        >>> a_location.move_y(-3).move_z(5)
        Location(frame=None, x=8.0, y=2.0, z=3.0)
        >>> a_location
        Location(frame=None, x=8.0, y=2.0, z=3.0)
        >>> a_location.move_by([-13, 3, -2])
        Location(frame=None, x=-5.0, y=5.0, z=1.0)

        >>> b_location = Location.from_scalars(x=3, y=8)
        >>> b_location
        Location(frame=None, x=3.0, y=8.0, z=0.0)
        >>> a_location + b_location
        Location(frame=None, x=-2.0, y=13.0, z=1.0)
        >>> a_location - b_location
        Location(frame=None, x=-8.0, y=-3.0, z=1.0)
        >>> -b_location
        Location(frame=None, x=-3.0, y=-8.0, z=0.0)
    """

    def __init__(
        self, frame: Pose | None = None, xyz: npt.ArrayLike = ZERO_VECTOR
    ) -> None:
        self._frame: Pose | None = frame
        self._v: FloatVector = np.asarray(xyz, dtype=float)

    @staticmethod
    def from_scalars(
        frame: Pose | None = None, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> Location:
        r"""Create a `Location` from scalar components.

        Returns:
            The new `Location` object.

        Examples:
            >>> a_location = Location.from_scalars()
            >>> a_location
            Location(frame=None, x=0.0, y=0.0, z=0.0)
            >>> print(a_location)
            (0.0, 0.0, 0.0)
            >>> print(a_location.frame)
            None

            >>> a_location = Location.from_scalars(None, 3.0, 5.0, -8.0)
            >>> a_location
            Location(frame=None, x=3.0, y=5.0, z=-8.0)
            >>> print(a_location)
            (3.0, 5.0, -8.0)
        """
        return Location(frame, (x, y, z))

    @property
    def frame(self) -> Pose | None:
        return self._frame

    # @frame.setter
    # def frame(self, frame: Pose) -> None:
    #     # FIXME: changing `frame` may imply updating internal representation
    #     self._frame = frame

    @property
    def x(self) -> float:
        return self._v[0]

    @x.setter
    def x(self, x: float) -> None:
        self._v[0] = x

    @property
    def y(self) -> float:
        return self._v[1]

    @y.setter
    def y(self, y: float) -> None:
        self._v[1] = y

    @property
    def z(self) -> float:
        return self._v[2]

    @z.setter
    def z(self, z: float) -> None:
        self._v[2] = z

    def __str__(self) -> str:
        return str(_round_tuple(self._v))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"frame={self.frame!s}, "
            f"x={_round_scalar(self.x)}, "
            f"y={_round_scalar(self.y)}, "
            f"z={_round_scalar(self.z)})"
        )

    def __add__(self, other: object) -> Location:
        if not isinstance(other, self.__class__):
            raise TypeError(f"{self.__class__.__name__} required.")
        if self._frame is not other._frame:
            raise ValueError("Locations must be in the same frame.")
        v = self._v + other._v
        return Location(self._frame, v)

    def __sub__(self, other: object) -> Location:
        if not isinstance(other, self.__class__):
            raise TypeError(f"{self.__class__.__name__} required.")
        if self._frame is not other._frame:
            raise ValueError("Locations must be in the same frame.")
        v = self._v - other._v
        return Location(self._frame, v)

    def __neg__(self) -> Location:
        v = -self._v
        return Location(self._frame, v)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and np.all(self._v == other._v)
        return False

    def approx_equal(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and (
                _round_tuple(self._v) == _round_tuple(other._v)
            )
        return False

    def copy(self) -> Location:
        """Create a copy of this `Location`.

        You can mutate the copy without changing the original.

        Returns:
            The new `Location` object.

        Examples:
            >>> a = Location.from_scalars(x=3.0, y=-5.0)
            >>> a
            Location(frame=None, x=3.0, y=-5.0, z=0.0)
            >>> b = a.copy()
            >>> b
            Location(frame=None, x=3.0, y=-5.0, z=0.0)
            >>> a is b
            False
            >>> a == b
            True

            >>> a.x = 2.2
            >>> b.z = -8.8
            >>> a
            Location(frame=None, x=2.2, y=-5.0, z=0.0)
            >>> b
            Location(frame=None, x=3.0, y=-5.0, z=-8.8)
            >>> a == b
            False
        """
        return Location(self._frame, self._v.copy())

    def move_to(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> Location:
        """Move to [_x_, _y_, _z_].

        Returns:
            This `Location`.
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        return self

    def move_x(self, dx: float) -> Location:
        """Move _x_ by _dx_ units.

        Returns:
            This `Location`.
        """
        self.x += dx
        return self

    def move_y(self, dy: float) -> Location:
        """Move _y_ by _dy_ units.

        Returns:
            This `Location`.
        """
        self.y += dy
        return self

    def move_z(self, dz: float) -> Location:
        """Move _z_ by _dz_ units.

        Returns:
            This `Location`.
        """
        self.z += dz
        return self

    def move_by(self, offset: npt.ArrayLike) -> Location:
        """Move by [_dx_, _dy_, _dz_] `offset`.

        Returns:
            This `Location`.
        """
        self._v += np.asarray(offset, dtype=float)
        return self

    def apply(self, vectors: npt.ArrayLike) -> FloatVector:
        """Apply this `Location` to translate _vectors_.

        Args:
            vectors: The `array([`_x_, _y_, _z_`])`
                or `array([[`_x_, _y_, _z_`], ...])` to translate.

        Returns:
            The translated `array([`_x'_, _y'_, _z'_`])`
            or `array([[`_x'_, _y'_, _z'_`], ...])`

        Examples:
            >>> an_offset = Location(None, [0, -1, 2])
            >>> an_offset
            Location(frame=None, x=0.0, y=-1.0, z=2.0)
            >>> an_offset.as_array()
            array([ 0., -1.,  2.])
            >>> an_offset.apply([3, 5, 8])
            array([ 3.,  4., 10.])

            >>> vectors = np.array([
            ...     [3, 5, 8],
            ...     [3, 5, -8],
            ...     [3, -5, -8],
            ...     [-3, -5, -8],
            ...     [-3, -5, 8],
            ...     [-3, 5, 8]
            ... ], dtype=float)
            >>> vectors
            array([[ 3.,  5.,  8.],
                   [ 3.,  5., -8.],
                   [ 3., -5., -8.],
                   [-3., -5., -8.],
                   [-3., -5.,  8.],
                   [-3.,  5.,  8.]])
            >>> an_offset.apply(vectors)
            array([[ 3.,  4., 10.],
                   [ 3.,  4., -6.],
                   [ 3., -6., -6.],
                   [-3., -6., -6.],
                   [-3., -6., 10.],
                   [-3.,  4., 10.]])
        """
        return vectors + self._v

    def inverse(self) -> Location:
        """Create a new `Location` that is the inverse of this `Location`.

        Returns:
            The new `Location` object.

        Examples:
            >>> fwd = Location.from_scalars(x=3, y=-5, z=8)
            >>> fwd
            Location(frame=None, x=3.0, y=-5.0, z=8.0)
            >>> inv = fwd.inverse()
            >>> inv
            Location(frame=None, x=-3.0, y=5.0, z=-8.0)
            >>> inv.inverse() == fwd
            True
        """
        v = -self._v
        return Location(self.frame, v)

    def in_frame(self, frame: Pose | None = None) -> Location:
        """Create a copy of this `Location` relative to another frame-of-reference.

        Returns:
            The new `Location` object.

        Args:
            frame: The `Pose` representing the target frame,
                or `None` for the world frame.

        Raises:
            ValueError:
                When this `Location` is not in the frame hierarchy.
        """
        origin: Pose | None = self.frame
        xyz: FloatVector = self._v
        while origin is not frame:
            if origin is None:
                raise ValueError("Location must be in the frame hierarchy.")
            xyz = origin.orientation.apply(xyz)
            offset: FloatVector = origin.location._v
            xyz += offset
            origin = origin.frame
        return Location(origin, xyz)

    def as_array(self) -> FloatVector:
        """This `Location` as an NDArray.

        This is an efficient accessor of internal state.
        **DO NOT MUTATE**

        Returns:
            `array([`_x_, _y_, _z_`])`
        """
        return self._v


IDENTITY_ROTATION: Rotation = Rotation.identity()
"""A neutral SciPy `Rotatiaon` object."""

IDENTITY_ROTATION_MATRIX: FloatVector = np.identity(3, dtype=float)
"""A 3x3 identity matrix for default Orientation."""


class Orientation:  # noqa: PLW1641
    r"""An orientation (rotation) in a given _reference frame_.

    Rotation angles are expressed in radians.

    Examples:
        >>> an_orientation = Orientation()
        >>> an_orientation
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        >>> print(an_orientation)
        (1.0, 0.0, 0.0, 0.0)
        >>> print(an_orientation.frame)
        None
        >>> an_orientation.w
        1.0
        >>> an_orientation.x
        0.0
        >>> an_orientation.y
        0.0
        >>> an_orientation.z
        0.0
        >>> an_orientation.as_array()
        array([1., 0., 0., 0.])

        >>> b_orientation = an_orientation.copy()
        >>> b_orientation
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        >>> an_orientation is b_orientation
        False
        >>> an_orientation == b_orientation
        True
        >>> an_orientation.pitch(np.pi/2)  # up 90°
        Orientation(frame=None, w=0.707107, x=0.707107, y=0.0, z=0.0)
        >>> b_orientation.yaw(-np.pi/2)  # right 90°
        Orientation(frame=None, w=0.707107, x=0.0, y=-0.707107, z=0.0)
        >>> an_orientation == b_orientation
        False

        >>> an_orientation.pitch(_deg(-90))  # down 90°
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        >>> an_orientation == Orientation()
        True

        >>> an_orientation.pitch(_deg(45)).yaw(_deg(90))  # up 45°, left 90°
        Orientation(frame=None, w=0.653281, x=0.270598, y=0.653281, z=0.270598)
        >>> an_orientation.yaw(_deg(-90)).pitch(_deg(-45))  # right 90°, down 45°
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
    """

    def __init__(
        self, frame: Pose | None = None, wxyz: npt.ArrayLike = IDENTITY_QUATERNION
    ) -> None:
        self._frame: Pose | None = frame
        wxyz = np.asarray(wxyz, dtype=float)
        self.__q: FloatVector = wxyz
        self.__r: Rotation | None = None
        self.__r_inv: Rotation | None = None

    @staticmethod
    def from_scalars(
        frame: Pose | None = None,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> Orientation:
        r"""Create an `Orientation` from scalar components.

        Returns:
            The new `Orientation` object.

        Examples:
            >>> an_orientation = Orientation.from_scalars()
            >>> an_orientation
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
            >>> print(an_orientation)
            (1.0, 0.0, 0.0, 0.0)
            >>> print(an_orientation.frame)
            None
            >>> an_orientation.yaw(_deg(30)).pitch(_deg(15))  # left 30°, up 15°
            Orientation(frame=None, w=0.957662, x=0.126079, y=0.256605, z=-0.033783)

            >>> b_orientation = Orientation.from_scalars(w=0.957662, x=0.126079, y=0.256605, z=-0.033783)
            >>> b_orientation
            Orientation(frame=None, w=0.957662, x=0.126079, y=0.256605, z=-0.033783)
            >>> print(b_orientation)
            (0.957662, 0.126079, 0.256605, -0.033783)

            >>> an_orientation == b_orientation
            False
            >>> an_orientation.approx_equal(b_orientation)
            True
            >>> print(b_orientation)
            (0.957662, 0.126079, 0.256605, -0.033783)
            >>> print(an_orientation)
            (0.957662, 0.126079, 0.256605, -0.033783)
            >>> an_orientation.as_array()
            array([ 0.9576622 ,  0.12607862,  0.25660481, -0.03378266])

            >>> b_orientation.pitch(_deg(-15)).yaw(_deg(-30))  # down 15°, right 30°
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        """  # noqa: E501
        return Orientation(frame, (w, x, y, z))

    @staticmethod
    def from_rotation(
        frame: Pose | None = None,
        r: Rotation = IDENTITY_ROTATION,
    ) -> Orientation:
        r"""Create an `Orientation` from a SciPy `Rotation`.

        Returns:
            The new `Orientation` object.

        Examples:
            >>> Orientation.from_rotation()
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)

            >>> r = Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
            >>> an_orientation = Orientation.from_rotation(None, r)
            >>> an_orientation
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
            >>> an_orientation.roll(_deg(-90))
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        """
        orientation = Orientation(frame)
        orientation._r = r  # cache Rotation property (sets `self._q`)
        return orientation

    @staticmethod
    def from_matrix(
        frame: Pose | None = None,
        matrix: npt.ArrayLike = IDENTITY_ROTATION_MATRIX,
    ) -> Orientation:
        r"""Create an `Orientation` from a rotation matrix.

        Returns:
            The new `Orientation` object.

        Examples:
            >>> Orientation.from_matrix()
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)

            >>> an_orientation = Orientation.from_matrix(None, [
            ...     [0, -1,  0],
            ...     [1,  0,  0],
            ...     [0,  0,  1]
            ... ])
            >>> an_orientation
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
            >>> an_orientation.roll(_deg(-90))
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        """
        r: Rotation = Rotation.from_matrix(matrix)
        return Orientation.from_rotation(frame, r)

    @property
    def frame(self) -> Pose | None:
        return self._frame

    # @frame.setter
    # def frame(self, frame: Pose) -> None:
    #     # FIXME: changing `frame` may imply updating internal representation
    #     self._frame = frame

    @property
    def _q(self) -> FloatVector:
        return self.__q

    @_q.setter
    def _q(self, q: FloatVector) -> None:
        self.__q = q
        self.__r = None
        self.__r_inv = None

    @property
    def _r(self) -> Rotation:
        if self.__r is None:
            wxyz: FloatVector = self._q
            xyzw = wxyz[..., [1, 2, 3, 0]]
            self.__r = Rotation.from_quat(xyzw)
        return self.__r

    @_r.setter
    def _r(self, r: Rotation) -> None:
        xyzw = r.as_quat()
        wxyz: FloatVector = xyzw[..., [3, 0, 1, 2]]
        self._q = wxyz
        self.__r = r
        self.__r_inv = None

    @property
    def _r_inv(self) -> Rotation:
        if self.__r_inv is None:
            self.__r_inv = self._r.inv()
        return self.__r_inv

    @property
    def w(self) -> float:
        return self._q[0]

    @property
    def x(self) -> float:
        return self._q[1]

    @property
    def y(self) -> float:
        return self._q[2]

    @property
    def z(self) -> float:
        return self._q[3]

    def __str__(self) -> str:
        return str(_round_tuple(self._q))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"frame={self.frame!s}, "
            f"w={_round_scalar(self.w)}, "
            f"x={_round_scalar(self.x)}, "
            f"y={_round_scalar(self.y)}, "
            f"z={_round_scalar(self.z)})"
        )

    def __mul__(self, other: object) -> Orientation:  # TODO: implement more operations?
        if not isinstance(other, self.__class__):
            raise TypeError(f"{self.__class__.__name__} required.")
        r = self._r * other._r
        xyzw: FloatVector = r.as_quat()
        wxyz = xyzw[..., [3, 0, 1, 2]]
        return Orientation(self.frame, wxyz)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and np.all(self._q == other._q)
        return False

    def approx_equal(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return (self.frame == other.frame) and (
                _round_tuple(self._q) == _round_tuple(other._q)
            )
        return False

    def copy(self) -> Orientation:
        """Create a copy of this `Orientation`.

        You can mutate the copy without changing the original.

        Returns:
            The new `Orientation` object.

        Examples:
            >>> a = Orientation()
            >>> a
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
            >>> a.pitch(_deg(180))
            Orientation(frame=None, w=0.0, x=1.0, y=0.0, z=0.0)

            >>> b = a.copy()
            >>> b
            Orientation(frame=None, w=0.0, x=1.0, y=0.0, z=0.0)
            >>> a is b
            False
            >>> a == b
            True

            >>> b.roll(_deg(90))
            Orientation(frame=None, w=0.0, x=0.707107, y=-0.707107, z=0.0)
            >>> a == b
            False
            >>> b.roll(_deg(-90))
            Orientation(frame=None, w=0.0, x=1.0, y=0.0, z=0.0)
            >>> a == b
            True
        """
        return Orientation(self.frame, self._q.copy())

    def pitch(self, d_phi: float) -> Orientation:
        """Rotate up by _phi_ radians (elevation).

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("x", d_phi)
        return self

    def yaw(self, d_theta: float) -> Orientation:
        """Rotate left by _theta_ radians (azimuth).

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("y", d_theta)
        return self

    def roll(self, d_psi: float) -> Orientation:
        """Rotate counter-clockwise by _psi_ radians.

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("z", d_psi)
        return self

    def apply(self, vectors: npt.ArrayLike) -> FloatVector:
        """Apply this `Orientation` to rotate _vectors_.

        Args:
            vectors: The `array([`_x_, _y_, _z_`])`
                or `array([[`_x_, _y_, _z_`], ...])` to rotate.

        Returns:
            The rotated `array([`_x'_, _y'_, _z'_`])`
            or `array([[`_x'_, _y'_, _z'_`], ...])`

        Examples:
            >>> a_rotation = Orientation()
            >>> a_rotation
            Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
            >>> a_rotation.as_array()
            array([1., 0., 0., 0.])
            >>> a_rotation.apply([3, 5, 8])
            array([3., 5., 8.])

            >>> a_rotation.pitch(_deg(180))
            Orientation(frame=None, w=0.0, x=1.0, y=0.0, z=0.0)
            >>> a_rotation.apply([3, 5, 8])
            array([ 3., -5., -8.])
            >>> a_rotation.roll(_deg(90))
            Orientation(frame=None, w=0.0, x=0.707107, y=-0.707107, z=0.0)
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
            >>> vectors
            array([[ 3.,  5.,  8.],
                   [ 3.,  5., -8.],
                   [ 3., -5., -8.],
                   [-3., -5., -8.],
                   [-3., -5.,  8.],
                   [-3.,  5.,  8.]])
            >>> a_rotation.apply(vectors)
            array([[-5., -3., -8.],
                   [-5., -3.,  8.],
                   [ 5., -3.,  8.],
                   [ 5.,  3.,  8.],
                   [ 5.,  3., -8.],
                   [-5.,  3., -8.]])
        """
        return self._r.apply(vectors)

    def inverse(self) -> Orientation:
        """Create a new `Orientation` that is the inverse of this `Orientation`.

        Returns:
            The new `Orientation` object.

        Examples:
            >>> fwd = Orientation()
            >>> fwd.yaw(-np.pi/2)  # right 90°
            Orientation(frame=None, w=0.707107, x=0.0, y=-0.707107, z=0.0)
            >>> inv = fwd.inverse()
            >>> inv
            Orientation(frame=None, w=-0.707107, x=0.0, y=-0.707107, z=0.0)
            >>> inv.inverse() == fwd
            True
        """
        return Orientation.from_rotation(self.frame, self._r_inv)

    def in_frame(self, frame: Pose | None = None) -> Orientation:
        """Create a copy of this `Orientation` relative to another frame-of-reference.

        Returns:
            The new `Orientation` object.

        Args:
            frame: The `Pose` representing the target frame,
                or `None` for the world frame.

        Raises:
            ValueError:
                When this `Orientation` is not in the frame hierarchy.
        """
        origin: Pose | None = self.frame
        r: Rotation = self._r
        while origin is not frame:
            if origin is None:
                raise ValueError("Orientation must be in the frame hierarchy.")
            rr: Rotation = origin.orientation._r
            r = rr * r
            origin = origin.frame
        return Orientation.from_rotation(origin, r)

    def rotation_to(self, target: Orientation) -> Orientation:
        """Calculate the rotation between this `Orientation` and the `target`.

        Returns:
            The rotation to `target`.

        Examples:
            >>> Orientation().yaw(_deg(-30))
            Orientation(frame=None, w=0.965926, x=0.0, y=-0.258819, z=0.0)
            >>> Orientation().yaw(_deg(-30)).inverse()
            Orientation(frame=None, w=-0.965926, x=0.0, y=-0.258819, z=0.0)
            >>> Orientation().yaw(_deg(30))
            Orientation(frame=None, w=0.965926, x=0.0, y=0.258819, z=0.0)
            >>> Orientation().yaw(_deg(30)).inverse()
            Orientation(frame=None, w=-0.965926, x=0.0, y=0.258819, z=0.0)

            >>> r0 = Orientation().yaw(_deg(45))
            >>> r1 = Orientation().yaw(_deg(15))
            >>> r2 = r0.rotation_to(r1)
            >>> r2
            Orientation(frame=None, w=-0.965926, x=0.0, y=0.258819, z=0.0)
            >>> r2.yaw(_deg(30))
            Orientation(frame=None, w=-1.0, x=0.0, y=0.0, z=0.0)

            >>> r1.rotation_to(r0)
            Orientation(frame=None, w=-0.965926, x=0.0, y=-0.258819, z=0.0)
            >>> r1.rotation_to(r0).to_matrix()
            array([[ 0.8660254,  0.       ,  0.5      ],
                   [-0.       ,  1.       ,  0.       ],
                   [-0.5      , -0.       ,  0.8660254]])
        """
        return target * self.inverse()

    def as_array(self) -> FloatVector:
        """This `Orientation` as an NDArray (quaternion).

        This is an efficient accessor of internal state.
        **DO NOT MUTATE**

        NOTE: The element order is scalar-first.

        Returns:
            `array([`_w_, _x_, _y_, _z_`])`
        """
        return self._q

    def to_rotation(self) -> Rotation:
        """Create a SciPy `Rotation` from this `Orientation`.

        Returns:
            A new `Rotation`

        Examples:
            >>> an_orientation = Orientation()
            >>> an_orientation.roll(np.pi/2)  # 90° around z-axis
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
            >>> r = an_orientation.to_rotation()
            >>> Orientation.from_rotation(None, r)
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
        """
        xyzw = self._r.as_quat()
        return Rotation.from_quat(xyzw)

    def to_matrix(self) -> npt.NDArray[Any]:  # FIXME: what should this return type be?
        """Create a rotation matrix from this `Orientation`.

        Returns:
            A 3x3 `NDArray`

        Examples:
            >>> an_orientation = Orientation()
            >>> an_orientation.roll(np.pi/2)  # 90° around z-axis
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
            >>> matrix = an_orientation.to_matrix()
            >>> Orientation.from_matrix(None, matrix)
            Orientation(frame=None, w=0.707107, x=0.0, y=0.0, z=0.707107)
        """
        return self._r.as_matrix()


class Pose:  # noqa: PLW1641
    r"""An object's location and orientation in a given _reference frame_.

    Each _reference frame_ is also a `Pose`.

    Examples:
        >>> world_frame = Pose(label="World")
        >>> world_frame
        Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
        >>> print(world_frame.frame)
        None

        >>> agent_location = world_frame.new_location([3, 5, -2])
        >>> agent_location
        Location(frame='World', x=3.0, y=5.0, z=-2.0)

        >>> agent_orientation = world_frame.new_orientation()
        >>> agent_orientation
        Orientation(frame='World', w=1.0, x=0.0, y=0.0, z=0.0)

        >>> agent_frame = world_frame.new_pose(agent_location, agent_orientation, 'Agent')
        >>> agent_frame
        Pose(frame='World', location=(3.0, 5.0, -2.0), orientation=(1.0, 0.0, 0.0, 0.0), label='Agent')

        >>> sensor_frame = agent_frame.new_frame((0.0, 1.5, 4.2), (0.707, 0.707, 0.0, 0.0))
        >>> sensor_frame
        Pose(frame='Agent', location=(0.0, 1.5, 4.2), orientation=(0.707, 0.707, 0.0, 0.0), label='')
        >>> sensor_frame.location
        Location(frame='Agent', x=0.0, y=1.5, z=4.2)
        >>> sensor_frame.orientation
        Orientation(frame='Agent', w=0.707, x=0.707, y=0.0, z=0.0)

        >>> sensor_frame.label = "Sensor"
        >>> s_point = sensor_frame.new_location([-7.4, 0.0, 4.7])
        >>> s_point
        Location(frame='Sensor', x=-7.4, y=0.0, z=4.7)
        >>> s_point.frame
        Pose(frame='Agent', location=(0.0, 1.5, 4.2), orientation=(0.707, 0.707, 0.0, 0.0), label='Sensor')
        >>> s_point.in_frame(sensor_frame)
        Location(frame='Sensor', x=-7.4, y=0.0, z=4.7)

        >>> a_point = s_point.in_frame(agent_frame)
        >>> a_point
        Location(frame='Agent', x=-7.4, y=-3.2, z=4.2)
        >>> a_point.frame
        Pose(frame='World', location=(3.0, 5.0, -2.0), orientation=(1.0, 0.0, 0.0, 0.0), label='Agent')

        >>> w_point = s_point.in_frame(world_frame)
        >>> w_point
        Location(frame='World', x=-4.4, y=1.8, z=2.2)
        >>> w_point.frame
        Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')

        >>> n_point = s_point.in_frame(None)
        >>> n_point
        Location(frame=None, x=-4.4, y=1.8, z=2.2)
        >>> n_point.frame is None
        True
    """  # noqa: E501

    def __init__(
        self,
        frame: Pose | None = None,
        location: Location | None = None,
        orientation: Orientation | None = None,
        label: str = "",
    ) -> None:
        self._frame: Pose | None = frame

        if location is None:
            location = Location(frame)
        if location.frame is not frame:
            raise ValueError(f"Location must be in frame {frame!s}.")
        self.location = location

        if orientation is None:
            orientation = Orientation(frame)
        if orientation.frame is not frame:
            raise ValueError(f"Orientation must be in frame {frame!s}.")
        self.orientation = orientation

        self.label: str = label

    @property
    def frame(self) -> Pose | None:
        return self._frame

    # @frame.setter
    # def frame(self, frame: Pose) -> None:
    #     # FIXME: changing `frame` may imply updating internal representation
    #     self._frame = frame

    def __str__(self) -> str:
        return f"{self.label!r}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"frame={self.frame!s}, "
            f"location={self.location}, "
            f"orientation={self.orientation}, "
            f"label={self.label!r})"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return (
                self.frame == other.frame
                and self.location == other.location
                and self.orientation == other.orientation
            )
        return False

    def copy(self) -> Pose:
        """Create a copy of this `Pose`.

        You can mutate the copy without changing the original.

        Returns:
            The new `Pose` object.

        Examples:
            >>> world_frame = Pose(label="World")
            >>> world_frame
            Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
            >>> global_frame = world_frame.copy()
            >>> world_frame is global_frame
            False
            >>> world_frame == global_frame
            True

            >>> global_frame.label = "Global"
            >>> global_frame
            Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='Global')
            >>> world_frame == global_frame
            True
        """  # noqa: E501
        return Pose(
            self.frame,
            self.location,
            self.orientation,
            self.label,
            # self.frame, self.location.copy(), self.orientation.copy(), self.label
        )

    def new_pose(
        self,
        location: Location | None = None,
        orientation: Orientation | None = None,
        label: str = "",
    ) -> Pose:
        return Pose(self, location, orientation, label)

    def new_frame(
        self,
        position: npt.ArrayLike = ZERO_VECTOR,
        rotation: npt.ArrayLike = IDENTITY_QUATERNION,
        label: str = "",
    ) -> Pose:
        return Pose(
            self,
            Location(self, position),
            Orientation(self, rotation),
            label,
        )

    def new_location(self, xyz: npt.ArrayLike = ZERO_VECTOR) -> Location:
        return Location(self, xyz)

    def new_orientation(self, wxyz: npt.ArrayLike = IDENTITY_QUATERNION) -> Orientation:
        return Orientation(self, wxyz)

    def inverse(self) -> Pose:
        """Create a new `Pose` that is the inverse of this `Pose`.

        Returns:
            The new `Pose` object.

        Examples:
            >>> world_frame = Pose(label="World")
            >>> world_frame
            Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
            >>> location = world_frame.new_location().move_to(x=5.0, y=8.0)
            >>> orientation = world_frame.new_orientation().roll(_deg(-30))
            >>> agent_in_world = world_frame.new_pose(location, orientation, "Agent")
            >>> agent_in_world
            Pose(frame='World', location=(5.0, 8.0, 0.0), orientation=(0.965926, 0.0, 0.0, -0.258819), label='Agent')

            >>> p = agent_in_world.new_location([-3.0, 2.0, -1.0])
            >>> p
            Location(frame='Agent', x=-3.0, y=2.0, z=-1.0)
            >>> p.in_frame(world_frame)
            Location(frame='World', x=3.401924, y=11.232051, z=-1.0)

            >>> world_in_agent = agent_in_world.inverse()
            >>> world_in_agent
            Pose(frame='Agent', location=(-0.330127, -9.428203, 0.0), orientation=(-0.965926, 0.0, 0.0, -0.258819), label='World')

            >>> p = world_in_agent.new_location([3.401924, 11.232051, -1.0])
            >>> p
            Location(frame='World', x=3.401924, y=11.232051, z=-1.0)
            >>> p.in_frame(agent_in_world)
            Location(frame='Agent', x=-3.0, y=2.0, z=-1.0)
        """  # noqa: E501
        orientation: Orientation = self.orientation.inverse()
        wxyz: FloatVector = orientation.as_array()
        location: Location = self.location.inverse()
        xyz: FloatVector = orientation.apply(location.as_array())
        label: str = "" if self.frame is None else self.frame.label
        return self.new_frame(xyz, wxyz, label)

    def in_frame(self, frame: Pose | None = None, label: str = "") -> Pose:
        """Create a copy of this `Pose` relative to another frame-of-reference.

        Returns:
            The new `Pose` object.

        Args:
            frame: The `Pose` representing the target frame,
                or `None` for the world frame.
            label: The label string displayed for this `Pose`.

        Examples:
            >>> agent_frame = Pose(label="Agent")
            >>> agent_frame.location.move_by([0.0, 0.13, -0.7])
            Location(frame=None, x=0.0, y=0.13, z=-0.7)
            >>> agent_frame.orientation.yaw(_deg(30))
            Orientation(frame=None, w=0.965926, x=0.0, y=0.258819, z=0.0)
            >>> agent_frame
            Pose(frame=None, location=(0.0, 0.13, -0.7), orientation=(0.965926, 0.0, 0.258819, 0.0), label='Agent')

            >>> sensor_frame = agent_frame.new_frame(label="Sensor")
            >>> sensor_frame.location.move_by([0.05, -0.03, 0.0])
            Location(frame='Agent', x=0.05, y=-0.03, z=0.0)
            >>> sensor_frame.orientation.pitch(_deg(-15))
            Orientation(frame='Agent', w=0.991445, x=-0.130526, y=0.0, z=0.0)
            >>> sensor_frame
            Pose(frame='Agent', location=(0.05, -0.03, 0.0), orientation=(0.991445, -0.130526, 0.0, 0.0), label='Sensor')

            >>> sensor_frame.in_frame(None)
            Pose(frame=None, location=(0.043301, 0.1, -0.725), orientation=(0.957662, -0.126079, 0.256605, 0.033783), label='')
        """  # noqa: E501
        location = self.location.in_frame(frame)
        orientation = self.orientation.in_frame(frame)
        return Pose(frame, location, orientation, label)
