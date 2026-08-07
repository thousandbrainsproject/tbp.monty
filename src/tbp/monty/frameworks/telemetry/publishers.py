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

from tbp.monty.frameworks.telemetry.schemas import TelemetryEvent


class TelemetryPublisher:
    """Structured telemetry publisher.

    Wraps a `logging.Logger` and emits `TelemetrySchema` as structured `LogRecord`
    instances routed through the logging pipeline to telemetry subscribers.

    Example::

        telemeter = TelemetryPublisher(__name__)
        telemeter.info(TelemetryEvent(...))
    """

    def __init__(self, name: str):
        """Creates a telemetry publisher backed by a named logger.

        The logger is namespaced under ``telemetry.*``.

        Args:
            name: Logger name suffix, usually the module name.
        """
        self.name = name
        self.event_logger = logging.getLogger(f"telemetry.{name}")
        self.event_logger.propagate = False  # do not propagate to root logger

        # Override NOTSET level to avoid delegation
        if self.event_logger.level == logging.NOTSET:
            self.event_logger.setLevel(logging.DEBUG)

        # Prevent logging.lastResort from printing to stderr
        if not self.event_logger.hasHandlers():
            self.event_logger.addHandler(logging.NullHandler())

    def emit(self, level: int, event: TelemetryEvent):
        """Emits a structured telemetry event at the specified log level.

        Args:
            level: The log level.
            event: The event instance.
        """
        self.event_logger.log(
            level=level,
            msg=event.kind,
            extra={"telemetry_schema": event},
            stacklevel=2,  # reports the stack frame that called this method
        )

    def debug(self, event: TelemetryEvent):
        """Emits a structured telemetry event at ``DEBUG`` log level."""
        self.emit(level=logging.DEBUG, event=event)

    def info(self, event: TelemetryEvent):
        """Emits a structured telemetry event at ``INFO`` log level."""
        self.emit(level=logging.INFO, event=event)

    def critical(self, event: TelemetryEvent):
        """Emits a structured telemetry event at ``CRITICAL`` log level."""
        self.emit(level=logging.CRITICAL, event=event)
