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
from typing import Callable, Iterable, Protocol, runtime_checkable

from tools.plot.interactive.generics import W
from tools.plot.interactive.topics import TopicMessage, TopicSpec


@runtime_checkable
class WidgetUpdaterProto(Protocol[W]):
    topics: Iterable[TopicSpec]

    def __call__(
        self, widget: W | None, msg: TopicMessage
    ) -> tuple[W | None, bool]: ...


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

    def __call__(self, widget: W | None, msg: TopicMessage) -> tuple[W | None, bool]:
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
