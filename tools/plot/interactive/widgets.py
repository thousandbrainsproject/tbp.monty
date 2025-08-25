# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from pubsub.core import Publisher
from vedo import Button, Slider2D

from tools.plot.interactive.utils import VtkDebounceScheduler


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


class WidgetOps(Protocol):
    """Protocol for defining specific widget state operations."""

    def extract_state(self, widget: Any) -> Any:
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

    def add(self, callback: Callable) -> Any:
        """Creates and returns a widget.

        Args:
            callback: Callable function for UI interaction.
            kwargs: additional kwargs to define how the widget will be added.

        Returns:
            The created widget object
        """
        ...

    def remove(self, widget: Any) -> None:
        """Removes a widget from a plotter.

        Args:
            widget: The widget to be removed
        """
        ...


class Widget:
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
        widget_ops: WidgetOps,
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

        self.widget = None
        self.state = None
        self.last_published_state = None
        self._sched_key = object()  # hashable unique key

        for topic in self.updater_topics:
            self.bus.subscribe(self._on_update_topic, topic)

    @property
    def updater_topics(self) -> set[str]:
        """Names of topics that can update this widget with `WidgetUpdater`.

        Returns:
            A set of topic names the widget listens to for updates.
        """
        if not hasattr(self.widget_ops, "updaters"):
            return {}

        return {t.name for u in self.widget_ops.updaters for t in u.topics}

    def add(self) -> None:
        """Create the widget and register the debounce callback.

        After creation, the wrapper schedules debounced publications using
        the shared scheduler.
        """
        add_fn = getattr(self.widget_ops, "add", None)
        self.widget = add_fn(self._on_change) if callable(add_fn) else None
        self.scheduler.register(self._sched_key, self._on_debounce_fire)

    def remove(self) -> None:
        """Remove the widget and cancel any pending debounced messages."""
        self.scheduler.cancel(self._sched_key)

        if self.widget is not None:
            self.widget_ops.remove(self.widget)
        self.widget = None

    def extract_state(self) -> Any:
        """Read the current state from the widget via `widget_ops`.

        Returns:
            The current state as defined by `widget_ops`.
        """
        extract_fn = getattr(self.widget_ops, "extract_state", None)
        return extract_fn(self.widget) if callable(extract_fn) else None

    def set_state(self, value: Any, publish: bool = True) -> None:
        """Set the widget state and optionally schedule a publish.

        Args:
            value: Desired state value.
            publish: If True, schedule a debounced publish.
        """
        self.widget_ops.set_state(self.widget, value)
        self.state = self.extract_state()

        if publish:
            self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_change(self, widget: Any, _event: str) -> None:
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
        if not hasattr(self.widget_ops, "updaters"):
            return

        for updater in self.widget_ops.updaters:
            self.widget, publish_state = updater(self.widget, msg)
            if publish_state:
                self.state = self.extract_state()
                self.scheduler.schedule_once(self._sched_key, self.debounce_sec)

    def _on_debounce_fire(self) -> None:
        """Handler fired by the scheduler to publish debounced state."""
        self._publish(self.extract_state())

    def _publish(self, state: Any) -> None:
        """Publish the state to the pubsub topic if not a duplicate.

        Args:
            state: State to publish.
        """
        if self.dedupe and self.last_published_state == state:
            return

        payload_fn = getattr(self.widget_ops, "state_to_messages", None)
        if not callable(payload_fn):
            return

        for msg in payload_fn(state):
            self.bus.sendMessage(msg.name, msg=msg)

        self.last_published_state = state


@dataclass
class TopicMessage:
    """Message passed on the pubsub bus.

    Attributes:
        name: Topic name.
        value: Value for the topic.
    """

    name: str
    value: Any


@dataclass
class TopicSpec:
    """Specification for a topic tracked by a widget updater.

    Attributes:
        name: Topic name to track.
        required: Whether this topic is required for the callback trigger. If
            True, the updater will not call the callback until a message for this
            topic arrives.
    """

    name: str
    required: bool = True


@dataclass
class WidgetUpdater:
    """Collect messages for a set of topics and callback when required topics are ready.

    The updater maintains an inbox keyed by topic name. Each time a message
    is received, it is recorded. When all required topics have at least one
    message, the callback is invoked with the current widget and the ordered
    inbox list.

    The callback decides how to update the widget and whether to publish
    the new state. It must return a tuple ``(widget, publish_state)``.

    Args:
        topics: Iterable of TopicSpec. Required topics gate readiness.
        callback: Called as callback(widget, inbox_list) when ready.
                  inbox_list is ordered by the topic spec order.

    Attributes:
        topics: Iterable of topic specs.
        callback: Callable that receives `(widget, inbox_list)` and returns
            `(widget, publish_state)`. The inbox list is ordered to match
            `topics`.
    """

    topics: Iterable[TopicSpec]
    callback: Callable

    _inbox: dict[str, TopicMessage] = field(default_factory=dict, init=False)

    @property
    def ready(self) -> bool:
        """Whether every required topic has at least one message."""
        return all(spec.name in self._inbox for spec in self.topics if spec.required)

    @property
    def inbox(self) -> list[TopicMessage]:
        """Inbox as a list ordered by the TopicSpec order, skipping missing ones."""
        return [
            self._inbox[spec.name] for spec in self.topics if spec.name in self._inbox
        ]

    def accepts(self, msg: TopicMessage) -> bool:
        """Check if this updater tracks the message's topic.

        Args:
            msg: Incoming topic message.

        Returns:
            True if the topic is listed in ``topics``. False otherwise.
        """
        return any(spec.name == msg.name for spec in self.topics)

    def __call__(self, widget: Any, msg: TopicMessage):
        """Record a message and invoke the callback if all required topics are ready.

        Args:
            widget: The widget instance to pass to the callback.
            msg: Received topic message.

        Returns:
            A tuple `(widget, publish_state)`. If the callback was invoked,
            this is whatever it returned. If not, returns `(widget, False)`.
        """
        if not self.accepts(msg):
            return widget, False

        self._inbox[msg.name] = msg

        if self.ready:
            return self.callback(widget, self.inbox)

        return widget, False
