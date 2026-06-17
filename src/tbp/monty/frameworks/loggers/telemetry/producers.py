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

from tbp.monty.frameworks.loggers.telemetry.events import TelemetryEvent


class TelemetryBroker(logging.Handler):
    """Fanout handler that routes telemetry events to subscribed consumer queues.

    Registered as a handler on the logging pipeline (typically via a QueueListener).
    On each emit(), looks up the event's schema_id and puts it onto every matching
    subscriber queue.

    Events are dispatched with `put_nowait()`. Full queues trigger `handleError()`
    rather than blocking the emitting thread.
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
            event_queue: Queue to receive matching events. If `None`, a new
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

        Extracts the `TelemetryEvent` from `record.monty_telemetry` and fans it out to
        each queue subscribed to its `schema_id`. Non-telemetry records are silently
        ignored.

        Args:
            record: The `LogRecord` emitted by `Telemetry.snapshot()`.
        """
        event: TelemetryEvent = record.__dict__.get("monty_telemetry")
        if not isinstance(event, TelemetryEvent):
            return  # TODO: log or raise error?
        with self.lock:
            event_queues = self.subscriptions.get(event.schema_id, [])
        for event_queue in event_queues:
            try:
                event_queue.put_nowait(event)
            except queue.Full:
                self.handleError(record)


class Telemetry:
    """Structured telemetry emitter, analogous to logging.Logger.

    Wraps a logging.Logger and emits TelemetryEvents as structured LogRecords
    routed through the logging pipeline to a TelemetryBroker. Obtain via a
    factory (e.g. `monty.get_telemetry(__name__)`) rather than instantiating
    directly.

    Usage::

        telemetry = monty.get_telemetry(__name__)
        telemetry.snapshot(logging.INFO, EpisodeResultEvent(...))
    """

    def __init__(self, logger: logging.Logger):
        """Telemetry constructor. Obtain via monty.get_telemetry() rather than directly.

        Args:
            logger: The underlying `logging.Logger`.
        """
        self._logger = logger

    def snapshot(self, level: int, event: TelemetryEvent):
        """Emit a structured telemetry event at the specified level.

        Args:
            level: logging level (`logging.DEBUG`, `logging.INFO`, etc.)
            event: The `TelemetryEvent` dataclass instance.
        """
        self._logger.log(
            level,
            event.schema_id,  # msg is the schema id for text sinks
            extra={"monty_telemetry": event},
            stacklevel=2,  # reports the stack frame that called `snapshot()`
        )
