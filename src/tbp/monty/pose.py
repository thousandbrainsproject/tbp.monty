# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Simple 3 and 4-tuples of `float`
from tbp.monty.math import QuaternionWXYZ, VectorXYZ

# A type alias for a 1D array of `float`
# FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatVector = np.ndarray


def _deg(d: float) -> float:
    """Convert degrees to radians.

    Radians | Degrees
    --------|--------
    Pi / 2  | 90
    Pi / 4  | 45
    Pi / 6  | 30
    Pi / 12 | 15
    Pi / 36 | 5

    Returns:
        The angle in radians
    """
    return d * np.pi / 180


def _round_scalar(n: float, ndigits: int = 6) -> float:
    n = round(n, ndigits)
    return 0.0 if n == 0.0 else n


def _round_tuple(t: tuple[float, ...], ndigits: int = 6) -> tuple[float, ...]:
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

        >>> a_location.x = 3
        >>> a_location.x
        3.0
        >>> a_location.y = 5
        >>> a_location.y
        5.0
        >>> a_location.z = -2
        >>> a_location.z
        -2.0
        >>> a_location.position
        (3.0, 5.0, -2.0)
        >>> a_location.to_xyz()
        (3.0, 5.0, -2.0)
        >>> a_location.to_vector()
        array([ 3.,  5., -2.])

        >>> a_location.delta_x(5)
        Location(frame=None, x=8.0, y=5.0, z=-2.0)
        >>> a_location.delta_y(-3).delta_z(5)
        Location(frame=None, x=8.0, y=2.0, z=3.0)
        >>> a_location
        Location(frame=None, x=8.0, y=2.0, z=3.0)
        >>> a_location.displace([-13, 3, -2])
        Location(frame=None, x=-5.0, y=5.0, z=1.0)

        >>> b_location = a_location.copy()
        >>> b_location
        Location(frame=None, x=-5.0, y=5.0, z=1.0)
        >>> a_location is b_location
        False
        >>> a_location == b_location
        True
        >>> a_location.x = 2.2
        >>> b_location.z = -3.3
        >>> a_location
        Location(frame=None, x=2.2, y=5.0, z=1.0)
        >>> b_location
        Location(frame=None, x=-5.0, y=5.0, z=-3.3)

        >>> a_location + b_location
        Location(frame=None, x=-2.8, y=10.0, z=-2.3)
        >>> a_location - b_location
        Location(frame=None, x=7.2, y=0.0, z=4.3)
    """

    def __init__(
        self, frame: Pose | None = None, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> None:
        self.frame: Pose | None = frame
        t: VectorXYZ = (float(x), float(y), float(z))
        self._v: FloatVector = np.array(t)

    def __str__(self) -> str:
        return str(_round_tuple(self.position))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"frame={self.frame!s}, "
            f"x={_round_scalar(self.x)}, "
            f"y={_round_scalar(self.y)}, "
            f"z={_round_scalar(self.z)})"
        )

    def __add__(self, other: object) -> Location:  # TODO: implement more operations?
        if not isinstance(other, self.__class__):
            raise TypeError(f"{self.__class__.__name__} required.")
        if self.frame is not other.frame:
            raise ValueError("Locations must be in the same frame.")
        v = self._v + other._v
        return Location(self.frame, *v)

    def __sub__(self, other: object) -> Location:
        if not isinstance(other, self.__class__):
            raise TypeError(f"{self.__class__.__name__} required.")
        if self.frame is not other.frame:
            raise ValueError("Locations must be in the same frame.")
        v = self._v - other._v
        return Location(self.frame, *v)

    def __neg__(self) -> Location:
        v = -self._v
        return Location(self.frame, *v)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and self.position == other.position
        return False

    def approx_equal(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and (
                _round_tuple(self.position) == _round_tuple(other.position)
            )
        return False

    def copy(self) -> Location:
        """Create a copy of this `Location`.

        You can mutate the copy without changing the original.

        Returns:
            The new `Location` object.
        """
        return Location(self.frame, self.x, self.y, self.z)

    @property
    def position(self) -> VectorXYZ:
        return self.to_xyz()

    @property
    def x(self) -> float:
        return self._v[0]

    @x.setter
    def x(self, x: float) -> None:
        self._v[0] = float(x)

    @property
    def y(self) -> float:
        return self._v[1]

    @y.setter
    def y(self, y: float) -> None:
        self._v[1] = float(y)

    @property
    def z(self) -> float:
        return self._v[2]

    @z.setter
    def z(self, z: float) -> None:
        self._v[2] = float(z)

    def delta_x(self, dx: float) -> Location:
        """Move _x_ by _dx_ units.

        Returns:
            This `Location`.
        """
        self.x += dx
        return self

    def delta_y(self, dy: float) -> Location:
        """Move _y_ by _dy_ units.

        Returns:
            This `Location`.
        """
        self.y += dy
        return self

    def delta_z(self, dz: float) -> Location:
        """Move _z_ by _dz_ units.

        Returns:
            This `Location`.
        """
        self.z += dz
        return self

    def displace(self, offset: FloatVector) -> Location:  # FIXME: choose a better name?
        """Move by (_dx_, _dy_, _dz_) 3-tuple `offset`.

        Returns:
            This `Location`.
        """
        self._v += offset
        return self

    def inverse(self) -> Location:
        """Create a new `Location` that is the inverse of this `Location`.

        Returns:
            The new `Location` object.
        """
        v = -self._v
        return Location(self.frame, *v)

    def in_frame(self, frame: Pose | None = None) -> Location:
        """Create a copy of this `Location` relative to another frame-of-reference.

        Returns:
            The new `Location` object.

        Raises:
            ValueError:
                When this `Location` is not in the frame hierarchy.
        """
        origin: Pose | None = self.frame
        # xyz: FloatVector = np.array(self.to_xyz())
        xyz: FloatVector = self._v
        while origin is not frame:
            if origin is None:
                raise ValueError("Location must be in the frame hierarchy.")
            xyz = origin.orientation.apply(xyz)
            # offset: FloatVector = np.array(origin.location.to_xyz())
            offset: FloatVector = origin.location._v
            xyz += offset
            origin = origin.frame
        return Location(origin, *xyz)

    def to_vector(self) -> FloatVector:
        """This `Location` as a vector.

        Returns:
            `array([`_x_, _y_, _z_`])`
        """
        return self._v.copy()

    def to_xyz(self) -> VectorXYZ:
        """This `Location` as an (_x_, _y_, _z_) tuple.

        Returns:
            `tuple(`_x_, _y_, _z_`)`
        """
        # FIXME: determine which of these implementations is more efficient
        # return tuple(self.x, self.y, self.z)
        return tuple(self._v)


class Orientation:  # noqa: PLW1641
    r"""An orientation (rotation) in a given _reference frame_.

    Rotation amounts are expressed in radians.

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
        >>> an_orientation.rotation
        (1.0, 0.0, 0.0, 0.0)
        >>> an_orientation.to_wxyz()
        (1.0, 0.0, 0.0, 0.0)
        >>> an_orientation.to_xyzw()
        (0.0, 0.0, 0.0, 1.0)
        >>> an_orientation.to_quat()
        array([1., 0., 0., 0.])

        >>> b_orientation = an_orientation.copy()
        >>> b_orientation
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        >>> an_orientation is b_orientation
        False
        >>> an_orientation == b_orientation
        False
        >>> an_orientation.approx_equal(b_orientation)
        True
        >>> _ = an_orientation.delta_pitch(np.pi/2)  # up 90°
        >>> _ = b_orientation.delta_yaw(-np.pi/2)  # right 90°
        >>> _round_tuple(an_orientation.rotation)
        (0.707107, 0.707107, 0.0, 0.0)
        >>> _round_tuple(b_orientation.rotation)
        (0.707107, 0.0, -0.707107, 0.0)

        >>> an_orientation.delta_pitch(-np.pi/2)  # down 90°
        Orientation(frame=None, w=1.0, x=0.0, y=0.0, z=0.0)
        >>> _round_tuple(an_orientation.rotation) == (1.0, 0.0, 0.0, 0.0)
        True
        >>> _ = an_orientation.delta_pitch(_deg(45)).delta_yaw(_deg(90))  # up 45°, left 90°
        >>> _round_tuple(an_orientation.rotation) == (1.0, 0.0, 0.0, 0.0)
        False
        >>> _ = an_orientation.delta_yaw(_deg(-90)).delta_pitch(_deg(-45))  # right 90°, down 45°
        >>> _round_tuple(an_orientation.rotation) == (1.0, 0.0, 0.0, 0.0)
        True

        >>> b_orientation
        Orientation(frame=None, w=0.707107, x=0.0, y=-0.707107, z=0.0)
        >>> b_orientation.inverse()
        Orientation(frame=None, w=-0.707107, x=0.0, y=-0.707107, z=0.0)
        >>> b_orientation.inverse().inverse()
        Orientation(frame=None, w=0.707107, x=0.0, y=-0.707107, z=0.0)
    """  # noqa: E501

    def __init__(
        self,
        frame: Pose | None = None,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        self.frame: Pose | None = frame
        self._r: Rotation = Rotation.from_quat([x, y, z, w])

    def __str__(self) -> str:
        return str(_round_tuple(self.rotation))

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
        return Orientation(self.frame, xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return self.frame == other.frame and self._r == other._r
        return False

    def approx_equal(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, self.__class__):
            # return self.frame == other.frame and self._r.approx_equal(other._r)
            return self.frame == other.frame and (
                _round_tuple(self.rotation) == _round_tuple(other.rotation)
            )
        return False

    def copy(self) -> Orientation:
        """Create a copy of this `Orientation`.

        You can mutate the copy without changing the original.

        Returns:
            The new `Orientation` object.
        """
        return Orientation(self.frame, self.w, self.x, self.y, self.z)

    @property
    def rotation(self) -> QuaternionWXYZ:
        return self.to_wxyz()

    @property
    def w(self) -> float:
        xyzw = self.to_xyzw()
        return xyzw[3]

    @property
    def x(self) -> float:
        xyzw = self.to_xyzw()
        return xyzw[0]

    @property
    def y(self) -> float:
        xyzw = self.to_xyzw()
        return xyzw[1]

    @property
    def z(self) -> float:
        xyzw = self.to_xyzw()
        return xyzw[2]

    def delta_pitch(self, d_phi: float) -> Orientation:
        """Rotate up by _phi_ radians (elevation).

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("x", d_phi)
        return self

    def delta_roll(self, d_psi: float) -> Orientation:
        """Rotate counter-clockwise by _psi_ radians.

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("z", d_psi)
        return self

    def delta_yaw(self, d_theta: float) -> Orientation:
        """Rotate left by _theta_ radians (azimuth).

        Returns:
            This `Orientation`.
        """
        self._r *= Rotation.from_euler("y", d_theta)
        return self

    def apply(self, vector: FloatVector) -> FloatVector:
        """Apply this `Orientation` to a _vector_.

        Returns:
            The rotated `array([`_x'_, _y'_, _z'_`])`
        """
        return self._r.apply(vector)

    def inverse(self) -> Orientation:
        """Create a new `Orientation` that is the inverse of this `Orientation`.

        Returns:
            The new `Orientation` object.
        """
        r: Rotation = self._r.inv()
        xyzw: FloatVector = r.as_quat()
        return Orientation(self.frame, xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def in_frame(self, frame: Pose = None) -> Orientation:
        """Create a copy of this `Orientation` relative to another frame-of-reference.

        Returns:
            The new `Orientation` object.

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
        xyzw: FloatVector = r.as_quat()
        return Orientation(origin, xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    def to_quat(self) -> FloatVector:
        """This `Orientation` as a quaternion.

        **WARNING!** The element order is scalar-first.

        Returns:
            `array([`_w_, _x_, _y_, _z_`])`
        """
        return np.array(self.to_wxyz())

    def to_wxyz(self) -> QuaternionWXYZ:
        """This `Orientation` as an (_w_, _x_, _y_, _z_) tuple.

        Returns:
            `tuple(`_w_, _x_, _y_, _z_`)`
        """
        xyzw = self._r.as_quat()
        wxyz: QuaternionWXYZ = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        return wxyz

    def to_xyzw(self) -> tuple[float, float, float, float]:
        """This `Orientation` as an (_x_, _y_, _z_, _w_) tuple.

        Returns:
            `tuple(`_x_, _y_, _z_, _w_`)`
        """
        xyzw = self._r.as_quat()
        return tuple(xyzw)


class Pose:  # noqa: PLW1641
    r"""An object's location and orientation in a given _reference frame_.

    Each _reference frame_ is also a `Pose`.

    Examples:
        >>> world_frame = Pose(label="World")
        >>> world_frame
        Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
        >>> print(world_frame.frame)
        None

        >>> global_frame = world_frame.copy()
        >>> global_frame
        Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
        >>> world_frame is global_frame
        False
        >>> world_frame == global_frame
        True
        >>> global_frame.label = "Global"
        >>> global_frame
        Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='Global')
        >>> world_frame == global_frame
        True

        >>> agent_location = world_frame.new_location(3, 5, -2)
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
        Pose(frame='Agent', location=(0.0, 1.5, 4.2), orientation=(0.707107, 0.707107, 0.0, 0.0), label='')
        >>> sensor_frame.location
        Location(frame='Agent', x=0.0, y=1.5, z=4.2)
        >>> sensor_frame.orientation
        Orientation(frame='Agent', w=0.707107, x=0.707107, y=0.0, z=0.0)

        >>> sensor_frame.label = "Sensor"
        >>> s_point = sensor_frame.new_location(-7.4, 0.0, 4.7)
        >>> s_point
        Location(frame='Sensor', x=-7.4, y=0.0, z=4.7)
        >>> s_point.frame
        Pose(frame='Agent', location=(0.0, 1.5, 4.2), orientation=(0.707107, 0.707107, 0.0, 0.0), label='Sensor')
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
        self.frame: Pose | None = frame

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
        return Pose(self.frame, self.location, self.orientation, self.label)

    def new_pose(
        self,
        location: Location | None = None,
        orientation: Orientation | None = None,
        label: str = "",
    ) -> Pose:
        return Pose(self, location, orientation, label)

    def new_frame(
        self,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        label: str = "",
    ) -> Pose:
        return Pose(
            self, Location(self, *position), Orientation(self, *rotation), label
        )

    def new_location(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Location:
        return Location(self, x, y, z)

    def new_orientation(
        self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> Orientation:
        return Orientation(self, w, x, y, z)

    def inverse(self) -> Pose:
        """Create a new `Pose` that is the inverse of this `Pose`.

        Returns:
            The new `Pose` object.

        Examples:
            >>> world_frame = Pose(label="World")
            >>> world_frame
            Pose(frame=None, location=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0), label='World')
            >>> location = world_frame.new_location(5.0, 8.0, 0.0)
            >>> orientation = world_frame.new_orientation().delta_roll(_deg(-30))
            >>> agent_in_world = world_frame.new_pose(location, orientation, "Agent")
            >>> agent_in_world
            Pose(frame='World', location=(5.0, 8.0, 0.0), orientation=(0.965926, 0.0, 0.0, -0.258819), label='Agent')

            >>> p = agent_in_world.new_location(-3.0, 2.0, -1.0)
            >>> p
            Location(frame='Agent', x=-3.0, y=2.0, z=-1.0)
            >>> p.in_frame(world_frame)
            Location(frame='World', x=3.401924, y=11.232051, z=-1.0)

            >>> world_in_agent = agent_in_world.inverse()
            >>> world_in_agent
            Pose(frame='Agent', location=(-0.330127, -9.428203, 0.0), orientation=(-0.965926, 0.0, 0.0, -0.258819), label='World')

            >>> p = world_in_agent.new_location(3.401924, 11.232051, -1.0)
            >>> p
            Location(frame='World', x=3.401924, y=11.232051, z=-1.0)
            >>> p.in_frame(agent_in_world)
            Location(frame='Agent', x=-3.0, y=2.0, z=-1.0)
        """  # noqa: E501
        orientation: Orientation = self.orientation.inverse()
        wxyz: QuaternionWXYZ = orientation.to_wxyz()
        location: Location = self.location.inverse()
        xyz: VectorXYZ = tuple(orientation.apply(location.to_vector()))
        label: str = "" if self.frame is None else self.frame.label
        return self.new_frame(xyz, wxyz, label)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
