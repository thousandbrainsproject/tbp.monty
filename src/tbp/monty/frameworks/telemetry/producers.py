# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import queue
from queue import Queue
from typing import Final

from tbp.monty.frameworks.telemetry.events import TelemetryEvent


class TelemetryBroker(logging.Handler):
    """Fanout handler that routes telemetry events to subscribed consumer queues.

    Registered as a handler on the logging pipeline (typically via a QueueListener).
    On each emit(), looks up the event's SCHEMA_ID and puts it onto every matching
    subscriber queue.

    Events are dispatched with ``put_nowait()``. Full queues trigger ``handleError()``
    rather than blocking the emitting thread.

    In practice, only a single broker exists (`TelemetryEmitter.broker` class var).
    """

    def __init__(self):
        """Initialize the broker with an empty subscription table."""
        super().__init__()
        self.subscriptions: dict[str, set[Queue[TelemetryEvent]]] = {}

    def subscribe(
        self,
        schema_ids: list[str],
        event_queue: Queue[TelemetryEvent] | None = None,
    ) -> Queue[TelemetryEvent]:
        """Registers a queue to receive events for the given schema IDs.

        The same queue may be subscribed to multiple schema IDs, in which case all
        matching events are multiplexed onto it. Thread-safe.

        Args:
            schema_ids: Schema ID strings to subscribe to.
            event_queue: Queue to receive matching events. If ``None``, a new
                         `queue.Queue` is created and returned.

        Returns:
            The subscribed queue.
        """
        if event_queue is None:
            event_queue = queue.Queue()
        with self.lock:
            for schema_id in schema_ids:
                self.subscriptions.setdefault(schema_id, set()).add(event_queue)
        return event_queue

    def unsubscribe(
        self,
        schema_ids: list[str],
        event_queue: Queue[TelemetryEvent],
    ):
        """Deregisters a queue from the given schema IDs. Thread-safe.

        Args:
            schema_ids: Schema IDs to unsubscribe from.
            event_queue: The queue to remove. Silently ignores unknown schema IDs.
        """
        # Force it into a list if it's a string, otherwise keep as is
        with self.lock:
            for schema_id in schema_ids:
                if schema_id in self.subscriptions:
                    self.subscriptions[schema_id].remove(event_queue)

    def emit(self, record: logging.LogRecord):
        """Dispatch a log record's telemetry event to all subscribed queues.

        Extracts the `TelemetryEvent` from ``record.telemetry_event`` and fans it out to
        each queue subscribed to its ``SCHEMA_ID``. Non-telemetry records are silently
        ignored. Overrides `logging.Handler.emit()`.

        Args:
            record: The `LogRecord` emitted by `Telemetry.snapshot()`.
        """
        event: TelemetryEvent = record.__dict__.get("telemetry_event")
        if not isinstance(event, TelemetryEvent):
            return  # TODO telemetry: log or raise error?
        with self.lock:
            event_queues = self.subscriptions.get(event.SCHEMA_ID, [])
        for event_queue in event_queues:
            try:
                event_queue.put_nowait(event)
            except queue.Full:
                self.handleError(record)


class TelemetryEmitter:
    """Structured telemetry emitter.

    Wraps a `logging.Logger` and emits `TelemetryEvent` as a structured `LogRecord`
    routed through the logging pipeline to a `TelemetryBroker`.

    Example::

        telemetry = TelemetryEmitter("name")
        telemetry.snapshot(logging.INFO, TelemetryEvent(...))
    """

    broker: Final = TelemetryBroker()
    """The main `TelemetryBroker` instance used globally throughout Monty."""

    def __init__(self, name, level: int):
        """Creates a telemetry emitter backed by a named logger.

        The logger is namespaced under ``monty_telemetry.*``, and prevents snapshot
        records from propagating to the root logger. All snapshot events are routed
        through a global `TelemetryBroker`, which forwards them to consumers based on
        their schema ID subscriptions.

        Args:
            name: Logger name for `logging`.
            level: Minimum logging level at which snapshots are emitted. Events below
                   this level are silently dropped before reaching the broker. If that
                   logger already exists, the new level overrides the old one.
        """
        self.name = name
        self.logger = logging.getLogger(f"monty_telemetry.{name}")
        self.logger.setLevel(level)
        self.logger.addHandler(self.broker)  # idempotent
        self.logger.propagate = False  # do not propagate events to root logger

    def snapshot(self, level: int, event: TelemetryEvent):
        """Emits a structured telemetry event at the specified logging level.

        Args:
            level: Logging level (``logging.DEBUG``, ``logging.INFO``, etc.)
            event: The `TelemetryEvent` instance.
        """
        self.logger.log(
            level,
            event.SCHEMA_ID,  # `msg` is the schema ID for text sinks
            extra={"telemetry_event": event},
            stacklevel=2,  # reports the stack frame that called `snapshot()`
        )
