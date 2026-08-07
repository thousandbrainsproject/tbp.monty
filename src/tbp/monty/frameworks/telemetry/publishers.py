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

from tbp.monty.frameworks import telemetry
from tbp.monty.frameworks.telemetry.schemas import TelemetryEvent, TelemetrySnapshot


class TelemetryPublisher:
    """Structured telemetry publisher.

    Wraps a `logging.Logger` and emits `TelemetrySchema` as structured `LogRecord`
    instances routed through the logging pipeline to telemetry subscribers.

    Example::

        telemeter = TelemetryPublisher(
            __name__,
            event_level=telemetry.INFO,
            snapshot_level=telemetry.TRACE
        )
        telemeter.emit(TelemetryEvent(...))
        telemeter.snapshot(TelemetrySnapshot(...))
    """

    def __init__(
        self,
        name: str,
        event_level=telemetry.NOTSET,
        snapshot_level=telemetry.NOTSET,
    ):
        """Creates a telemetry publisher backed by named loggers.

        Loggers are namespaced under ``telemetry.*`` and ``snapshots.*``. Subscribers
        self-register as handlers to specific loggers.

        Schemas emitted at levels below ``event_level`` or ``snapshot_level`` are
        silently dropped before reaching their respective handlers.

        If a logger already exists, and its level chosen here is different from
        ``telemetry.NOTSET`` (0), the new level overrides the old.

        Args:
            name: Logger name suffix.
            event_level: Minimum `telemetry` log level for the event logger.
            snapshot_level: Minimum `telemetry` log level for the snapshot logger.
        """
        self.name = name

        self.event_logger = logging.getLogger(f"telemetry.{name}")
        self.event_logger.propagate = False  # do not propagate to root logger
        if event_level != telemetry.NOTSET:
            self.event_logger.setLevel(event_level)

        # Prevent logging.lastResort from printing to stderr
        if not self.event_logger.hasHandlers():
            self.event_logger.addHandler(logging.NullHandler())

        self.snapshot_logger = logging.getLogger(f"snapshots.{name}")
        self.snapshot_logger.propagate = False
        if snapshot_level != telemetry.NOTSET:
            self.snapshot_logger.setLevel(snapshot_level)

        if not self.snapshot_logger.hasHandlers():
            self.snapshot_logger.addHandler(logging.NullHandler())

    def emit(self, event: TelemetryEvent, level=telemetry.INFO):
        """Emits a structured telemetry event at the specified log level.

        Args:
            event: The event instance.
            level: The `telemetry` log level.
        """
        self.event_logger.log(
            level=level,
            msg=event.kind,
            extra={"telemetry_schema": event},
            stacklevel=2,  # reports the stack frame that called this method
        )

    def snapshot(self, snapshot: TelemetrySnapshot):
        """Emits a structured telemetry snapshot at the ``telemetry.TRACE`` level.

        Args:
            snapshot: The snapshot instance.
        """
        self.snapshot_logger.log(
            level=telemetry.TRACE,
            msg=snapshot.kind,
            extra={"telemetry_schema": snapshot},
            stacklevel=2,  # reports the stack frame that called this method
        )
