# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any, Generic

from pubsub.core import Publisher
from vedo import Button, Slider2D

from tools.plot.interactive.generics import S, W
from tools.plot.interactive.topics import TopicMessage
from tools.plot.interactive.utils import VtkDebounceScheduler
from tools.plot.interactive.widget_ops import (
    HasStateToMessages,
    HasUpdaters,
    SupportsAdd,
    SupportsExtractState,
    SupportsRemove,
    SupportsSetState,
    WidgetOpsProto,
)


def extract_slider_state(widget: Slider2D) -> int:
    """Read the slider state and round it to an integer value.

    Args:
        widget: The Vedo slider.

    Returns:
        The current slider value rounded to the nearest integer.
    """
    return round(widget.GetRepresentation().GetValue())


def set_slider_state(widget: Slider2D, value: Any) -> None:
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


def set_button_state(widget: Button, value: str | int):
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


class Widget(Generic[W, S]):
    """High-level wrapper that connects a Vedo widget to a pubsub topic.

    The widget is created via `widget_ops.add` and removed via
    `widget_ops.remove`. State reads and writes are delegated to
    `widget_ops`. This wrapper implements Debounce logic through the
    `VtkDebounceScheduler`, which runs a timer in the background effectively
    collapsing rapid changes in widget states.

    Attributes:
        bus: Pubsub bus used to send messages.
        scheduler: Debounce scheduler used to collapse rapid UI changes.
        widget_ops: Composed functionality for get/set/add/remove operations.
        debounce_sec: Debounce delay in seconds for change publications.
        dedupe: If True, skip publishing unchanged values.

    Runtime Attributes:
        widget: The created widget instance.
        state: Last observed state value.
        last_published_state: Previous published state value for dedupe logic.
        _sched_key: Unique hashable key for the scheduler.
    """

    def __init__(
        self,
        widget_ops: (
            SupportsAdd[W]
            | SupportsRemove[W]
            | SupportsExtractState[W, S]
            | SupportsSetState[W, S]
            | HasStateToMessages[S]
            | HasUpdaters[W]
            | WidgetOpsProto
        ),
        bus: Publisher,
        scheduler: VtkDebounceScheduler,
        debounce_sec: float = 0.25,
        dedupe: bool = True,
    ):
        self.bus = bus
        self.scheduler = scheduler
        self.debounce_sec = debounce_sec
        self.dedupe = dedupe
        self.widget_ops = widget_ops

        self.widget: W | None = None
        self.state: S | None = None
        self.last_published_state: S | None = None
        self._sched_key = object()  # hashable unique key

        if isinstance(self.widget_ops, HasUpdaters):
            for topic in self.updater_topics:
                self.bus.subscribe(self._on_update_topic, topic)

    @property
    def updater_topics(self) -> set[str]:
        """Names of topics that can update this widget with `WidgetUpdater`.

        Returns:
            A set of topic names the widget listens to for updates.
        """
        if not isinstance(self.widget_ops, HasUpdaters):
            return set()

        return {t.name for u in self.widget_ops.updaters for t in u.topics}

    def add(self) -> None:
        """Create the widget and register the debounce callback.

        After creation, the wrapper schedules debounced publications using
        the shared scheduler.
        """
        if isinstance(self.widget_ops, SupportsAdd):
            self.widget = self.widget_ops.add(self._on_change)
        else:
            self.widget = None

        self.scheduler.register(self._sched_key, self._on_debounce_fire)

    def remove(self) -> None:
        """Remove the widget and cancel any pending debounced messages."""
        self.scheduler.cancel(self._sched_key)
        if self.widget is not None and isinstance(self.widget_ops, SupportsRemove):
            self.widget_ops.remove(self.widget)
        self.widget = None

    def extract_state(self) -> S | None:
        """Read the current state from the widget via `widget_ops`.

        Returns:
            The current state as defined by `widget_ops`.
        """
        if isinstance(self.widget_ops, SupportsExtractState):
            return self.widget_ops.extract_state(self.widget)
        return None

    def set_state(self, value: S, publish: bool = True) -> None:
        """Set the widget state and optionally schedule a publish.

        Args:
            value: Desired state value.
            publish: If True, schedule a debounced publish.
        """
        if not isinstance(self.widget_ops, SupportsSetState):
            raise NotImplementedError("This WidgetOps does not support set_state.")

        self.widget_ops.set_state(self.widget, value)
        self.state = self.extract_state()
        if publish:
            self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_change(self, widget: W, _event: str) -> None:
        """Internal callback when the underlying widget reports a UI change.

        When a widget value changes from the UI (e.g., slider moved or button
        pressed), this function gets called, which extracts the new state and
        publishes it.

        Args:
            widget: The VTK widget instance.
            _event: Event name from VTK/Vedo.
        """
        if isinstance(widget, Button):
            widget.switch()

        self.state = self.extract_state()
        self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_update_topic(self, msg: TopicMessage):
        if not isinstance(self.widget_ops, HasUpdaters):
            return

        for updater in self.widget_ops.updaters:
            self.widget, publish_state = updater(self.widget, msg)
            if publish_state:
                self.state = self.extract_state()
                self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_debounce_fire(self) -> None:
        """Handler fired by the scheduler to publish debounced state."""
        self._publish(self.extract_state())

    def _publish(self, state: S | None) -> None:
        """Publish the state to the pubsub topic if not a duplicate.

        Args:
            state: State to publish.
        """
        if self.dedupe and self.last_published_state == state:
            return

        if isinstance(self.widget_ops, HasStateToMessages):
            for msg in self.widget_ops.state_to_messages(state):
                self.bus.sendMessage(msg.name, msg=msg)

            self.last_published_state = state
