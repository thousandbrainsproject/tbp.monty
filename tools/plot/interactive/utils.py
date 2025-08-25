# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Hashable

import numpy as np
import numpy.typing as npt
from vedo.vtkclasses import vtkRenderWindowInteractor


@dataclass
class Location2D:
    """2D location wrapper.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
    """

    x: float
    y: float

    def __eq__(self, other: object) -> bool:
        """Return True only if same class and all values match exactly."""
        if other.__class__ is self.__class__:
            return self.x == other.x and self.y == other.y
        return False

    def to_3d(self, z: float) -> Location3D:
        """Create a 3D location by adding a z coordinate.

        Args:
            z: Z coordinate value.

        Returns:
            A new Location3D with (x, y, z).
        """
        return Location3D(self.x, self.y, z)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert to a NumPy array in the form [x, y].

        Returns:
            A NumPy array with shape (2,).
        """
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Location3D:
    """3D location wrapper.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
    """

    x: float
    y: float
    z: float

    def __eq__(self, other: object) -> bool:
        """Return True only if same class and all values match exactly."""
        if other.__class__ is self.__class__:
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def to_2d(self) -> Location2D:
        """Drop the z coordinate and return a 2D location.

        Returns:
            A new Location2D with (x, y).
        """
        return Location2D(self.x, self.y)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert to a NumPy array in the form [x, y, z].

        Returns:
            A NumPy array with shape (3,).
        """
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Bounds:
    """rectangle bounds described by (xmin, xmax, ymin, ymax).

    Attributes:
        xmin: Minimum x value.
        xmax: Maximum x value.
        ymin: Minimum y value.
        ymax: Maximum y value.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

    def contains(self, location: Location2D) -> bool:
        """Check whether a 2D point lies inside or on the boundary.

        Args:
            location: Location to test.

        Returns:
            True if x and y values lie within the rectangle boundary.
        """
        return (self.xmin <= location.x <= self.xmax) and (
            self.ymin <= location.y <= self.ymax
        )


@dataclass
class CoordinateMapper:
    """Map between GUI click coordinates and data coordinates, and back.

    Attributes:
        gui: Bounds for the GUI rectangle.
        data: Bounds for the data rectangle.

    Raises:
        ValueError: If any provided bounds have nonpositive width or height.
    """

    gui: Bounds
    data: Bounds

    def __post_init__(self) -> None:
        if self.gui.width() <= 0 or self.gui.height() <= 0:
            raise ValueError("GUI bounds must have positive width and height")
        if self.data.width() <= 0 or self.data.height() <= 0:
            raise ValueError("Data bounds must have positive width and height")

    def map_click_to_data_coords(self, location: Location2D) -> Location2D:
        """Map a GUI click (x, y) to data coordinates (x_val, y_val).

        Args:
            location: Click location in GUI coordinates.

        Returns:
            Location of the click point in data coordinates.
        """
        x_rel = (location.x - self.gui.xmin) / self.gui.width()
        y_rel = (location.y - self.gui.ymin) / self.gui.height()

        x_val = self.data.xmin + x_rel * self.data.width()
        y_val = self.data.ymin + y_rel * self.data.height()
        return Location2D(x_val, y_val)

    def map_data_coords_to_world(self, location: Location2D) -> Location2D:
        """Map data coordinates (x, y) back to GUI coordinates (x_gui, y_gui).

        Args:
            location: Point in data coordinates.

        Returns:
            Location of the point in GUI coordinates.
        """
        x_rel = (location.x - self.data.xmin) / self.data.width()
        y_rel = (location.y - self.data.ymin) / self.data.height()

        x_gui = self.gui.xmin + x_rel * self.gui.width()
        y_gui = self.gui.ymin + y_rel * self.gui.height()
        return Location2D(x_gui, y_gui)


class VtkDebounceScheduler:
    """Single repeating VTK timer that services many debounced callbacks.

    The scheduler keeps one repeating VTK timer and a registry of callbacks that
    are scheduled to run once at or after a given time. Each callback is keyed
    by a hashable token.

    Attributes:
        _iren: A `vtkRenderWindowInteractor` object.
        _period_ms: Timer period in milliseconds.
        _obs_tag: Observer tag for the registered VTK timer event.
        _timer_id: VTK timer id.
        _callbacks: Mapping from keys to callbacks.
        _due: Mapping from keys to ready times in seconds.
    """

    def __init__(self, interactor: vtkRenderWindowInteractor, period_ms: int = 33):
        """Initialize the scheduler.

        Args:
            interactor: VTK render window interactor.
            period_ms: Repeating timer period in milliseconds.
        """
        self._iren = interactor
        self._period_ms = period_ms

        self._obs_tag: int | None = None
        self._timer_id: int | None = None
        self._callbacks: dict[Hashable, Callable[[], None]] = {}
        self._due: dict[Hashable, float] = {}

    def start(self) -> None:
        """Ensure the repeating timer is running and the observer is set."""
        if self._obs_tag is None:
            self._obs_tag = self._iren.AddObserver("TimerEvent", self._on_timer)
        if self._timer_id is None:
            self._timer_id = self._iren.CreateRepeatingTimer(self._period_ms)

    def register(self, key: Hashable, callback: Callable[[], None]) -> None:
        """Register a callback under a key and start the timer if needed.

        Args:
            key: Unique hashable key for the callback.
            callback: callback function to invoke when due.
        """
        self._callbacks[key] = callback
        self.start()

    def schedule_once(self, key: Hashable, delay_sec: float) -> None:
        """Schedule a registered callback to run after a delay.

        Args:
            key: Key of a previously registered callback.
            delay_sec: Delay in seconds. If less than or equal to zero, schedule
                immediately.

        Raises:
            KeyError: If the key is not registered.
        """
        if key not in self._callbacks:
            raise KeyError("Key not registered with scheduler")
        now = time.perf_counter()
        self._due[key] = now if delay_sec <= 0 else now + delay_sec

    def cancel(self, key: Hashable) -> None:
        """Cancel a scheduled callback and remove it from the registry.

        Args:
            key: Key for the callback to cancel.
        """
        self._due.pop(key, None)
        self._callbacks.pop(key, None)
        if not self._callbacks:
            self._teardown()

    def shutdown(self) -> None:
        """Clear all callbacks and tear down the timer and observer."""
        self._due.clear()
        self._callbacks.clear()
        self._teardown()

    def _teardown(self) -> None:
        """Tear down the VTK timer and observer if present."""
        if self._timer_id is not None:
            with suppress(Exception):
                self._iren.DestroyTimer(self._timer_id)
            self._timer_id = None
        if self._obs_tag is not None:
            with suppress(Exception):
                self._iren.RemoveObserver(self._obs_tag)
            self._obs_tag = None

    def _on_timer(self, _obj: Any, _evt: str) -> None:
        """VTK timer event handler.

        Args:
            _obj: VTK callback object (i.e., vtkXRenderWindowInteractor).
            _evt: Event name (e.g., "TimerEvent").
        """
        if not self._due:
            return
        now = time.perf_counter()
        ready = [k for k, t in list(self._due.items()) if now >= t]
        for key in ready:
            self._due.pop(key, None)
            cb = self._callbacks.get(key)
            if cb:
                cb()
