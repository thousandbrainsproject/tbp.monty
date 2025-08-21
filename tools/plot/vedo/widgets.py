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
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional, Protocol

from pubsub.core import Publisher
from vedo import Button, Plotter
from vedo.vtkclasses import vtkRenderWindowInteractor


class WidgetStateOps(Protocol):
    """Protocol for defining specific widget state operations."""

    def get_state(self, widget: Any) -> Any:
        """Return the current state of a widget.

        Args:
            widget: The underlying VTK/Vedo widget instance.

        Returns:
            The current state of the widget.
        """
        ...

    def set_state(self, widget: Any, value: Any) -> None:
        """Set the state of a widget.

        Args:
            widget: The underlying VTK/Vedo widget instance.
            value: Desired state value, type depends on widget.
        """
        ...

    def add_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to add this widget to a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes the proper add function.

        Returns:
            A callable that creates the widget when invoked.
        """
        ...

    def remove_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to remove this widget from a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes a `remove` method.

        Returns:
            A callable that removes a given widget when invoked.
        """
        ...


class SliderStateOps:
    """State operations for a Vedo slider widget."""

    def get_state(self, widget: Any) -> int:
        """Read the slider state and round it to an integer value.

        Args:
            widget: The Vedo slider.

        Returns:
            The current slider value rounded to the nearest integer.
        """
        return round(widget.GetRepresentation().GetValue())

    def set_state(self, widget: Any, value: Any) -> None:
        """Set the slider value after type and range checks.

        Args:
            widget: The Vedo slider.
            value: The requested value to set.

        Raises:
            TypeError: If value cannot be converted to float.
            ValueError: If value falls outside `widget.range`.
        """
        try:
            value = float(value)
        except (TypeError, ValueError) as err:
            raise TypeError("Slider value must be castable to float") from err

        min_val, max_val = widget.range
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"Slider requested value {value} out of range [{min_val}, {max_val}]"
            )

        widget.GetRepresentation().SetValue(float(value))

    def add_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to add this slider to a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes the `add_slider` function.

        Returns:
            A callable that creates the widget when invoked.
        """
        return plotter.add_slider

    def remove_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to remove this Slider from a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes a `remove` method.

        Returns:
            A callable that removes a given widget when invoked.
        """
        return plotter.remove


class ButtonStateOps:
    """State operations for a Vedo button with discrete `states`."""

    def get_state(self, widget: Any) -> str:
        """Read the current button state label.

        Args:
            widget: The Vedo button.

        Returns:
            The current state string.
        """
        return widget.states[widget.status_idx]

    def set_state(self, widget: Any, value: Any) -> None:
        """Set the button state by label or index.

        Args:
            widget: The Vedo button.
            value: Either a string in `widget.states` or an int index.

        Raises:
            TypeError: If value is neither int nor str.
            ValueError: If index is out of range or label is unknown.
        """
        states = list(widget.states)

        if isinstance(value, str):
            try:
                idx = states.index(value)
            except ValueError as err:
                raise ValueError(
                    f"Unknown state {value!r}. Allowed states: {states}"
                ) from err
        elif isinstance(value, int):
            if not 0 <= value < len(states):
                raise ValueError(f"Index {value} out of range")
            idx = value
        else:
            raise TypeError("value must be int or str")

        widget.status_idx = idx
        widget.status(idx)

    def add_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to add this button to a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes the `add_button` function.

        Returns:
            A callable that creates the widget when invoked.
        """
        return plotter.add_button

    def remove_fn(self, plotter: Plotter) -> Callable:
        """Return the function used to remove this button from a plotter.

        Args:
            plotter: A `vedo.Plotter` that exposes a `remove` method.

        Returns:
            A callable that removes a given widget when invoked.
        """
        return plotter.remove


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


@dataclass
class Widget:
    """High-level wrapper that connects a Vedo widget to a pubsub topic.

    The widget is created via `state_ops.add_fn(plotter)` and removed via
    `state_ops.remove_fn(plotter)`. State reads and writes are delegated to
    `state_ops`. This wrapper implements Debounce logic through the
    `VtkDebounceScheduler`, which runs a timer in the background effectively
    collapsing rapid changes in widget states.

    Note that the widget-specific operations are extracted to the a composed
    class that follows the `WidgetStateOps` protocol (e.g., `SliderStateOps`,
    `ButtonStateOps`).

    Attributes:
        topic: Pubsub topic name to publish state on.
        bus: Pubsub bus used to send messages.
        scheduler: Debounce scheduler used to collapse rapid UI changes.
        state_ops: Strategy object for get/set/add/remove operations.
        plotter: A `vedo.Plotter` host to attach widgets to.
        debounce_sec: Debounce delay in seconds for change publications.
        dedupe: If True, skip publishing unchanged values.
        payload_fn: Optional function that maps a state to a payload dict.
        add_kwargs: Keyword arguments forwarded to the underlying add function.

    Runtime Attributes:
        widget: The created widget instance.
        state: Last observed state value.
        last_published_state: Previous published state value for dedupe logic.
        _sched_key: Unique hashable key for the scheduler.
    """

    topic: str
    bus: Publisher
    scheduler: VtkDebounceScheduler
    state_ops: WidgetStateOps
    plotter: Plotter
    debounce_sec: float = 0.25
    dedupe: bool = True
    payload_fn: Optional[Callable[[Any], dict]] = None
    add_kwargs: dict[str, Any] = field(default_factory=dict)

    # runtime
    widget: Any | None = field(init=False, default=None)
    state: Any = field(init=False, default=None)
    last_published_state: Any = field(init=False, default=None)
    _sched_key: object = field(init=False, default_factory=object)

    def add(self) -> None:
        """Create the widget and register debounce."""
        add_fn = self.state_ops.add_fn(self.plotter)
        self.widget = add_fn(self._on_change, **self.add_kwargs)
        self.state = self._extract_state()
        self.scheduler.register(self._sched_key, self._on_debounce_fire)

    def remove(self, publish_latest: bool = False) -> None:
        """Remove the widget from the plotter.

        Args:
            publish_latest: If True, publish the latest state before removal.
        """
        if publish_latest and self.widget is not None:
            self._publish(self._extract_state(), force=False)

        self.scheduler.cancel(self._sched_key)
        if self.widget is not None:
            remove_fn = self.state_ops.remove_fn(self.plotter)
            remove_fn(self.widget)

        self.widget = None

    def set_state(self, value: Any, publish: bool = True) -> None:
        """Set the widget state and optionally schedule a publish.

        Args:
            value: Desired state value.
            publish: If True, schedule a debounced publish.
        """
        if not self.widget:
            return

        self.state_ops.set_state(self.widget, value)
        self.state = self._extract_state()

        if publish:
            self._schedule_once()

    def force_publish(self) -> None:
        """Publish the current state immediately, bypassing debounce and dedupe."""
        self._publish(self._extract_state(), force=True)

    def _on_change(self, widget: Any, _event: str) -> None:
        """Internal callback when the underlying widget reports a UI change.

        Args:
            widget: The VTK widget instance.
            _event: Event name from VTK/Vedo.
        """
        if isinstance(widget, Button):
            widget.switch()

        self.state = self._extract_state()
        self._schedule_once()

    def _schedule_once(self) -> None:
        """Schedule a debounced publish."""
        self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_debounce_fire(self) -> None:
        """Handler fired by the scheduler to publish debounced state."""
        self._publish(self._extract_state(), force=False)

    def _extract_state(self) -> Any:
        """Read the current state from the widget via `state_ops`.

        Returns:
            The current state or None if the widget is not present.
        """
        if not self.widget:
            return None

        return self.state_ops.get_state(self.widget)

    def _payload_for(self, state: Any) -> dict:
        """Build the message payload for a given state.

        Args:
            state: State value to serialize.

        Returns:
            A dictionary payload to send on the pubsub bus.
        """
        if self.payload_fn:
            return self.payload_fn(state)
        return {"value": state, "ts": time.time()}

    def _publish(self, state: Any, force: bool) -> None:
        """Publish the state to the pubsub topic if not a duplicate or if forced.

        Args:
            state: State to publish.
            force: If True, publish regardless of dedupe checks.
        """
        if self.dedupe and not force and self.last_published_state == state:
            return
        self.bus.sendMessage(self.topic, **self._payload_for(state))
        self.last_published_state = state
