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
from typing import NoReturn

from typing_extensions import deprecated

from tbp.monty.telemetry.schemas import TelemetryEvent


class TelemetryPublisher(logging.Logger):
    """Structured telemetry publisher.

    Subclasses `logging.Logger` and emits `TelemetrySchema` as structured `LogRecord`
    instances routed through the logging pipeline to telemetry handlers.

    Do not instantiate this class directly; obtain it via `telemetry.getTelemeter`.

    Example::

        telemeter = telemetry.getTelemeter(__name__)
        telemeter.info(TelemetryEvent(...))
    """

    def __init__(self, *args, **kwargs):
        """Initializes the logger; do not instantiate this class outside its module."""
        super().__init__(*args, **kwargs)
        self.propagate = False  # do not propagate to root logger

        # Prevent logging.lastResort from printing to stderr
        if not self.hasHandlers():
            self.addHandler(logging.NullHandler())

    def log(self, level: int, event: TelemetryEvent, *args, **kwargs):
        """Emits a structured telemetry event at the specified log level.

        Args:
            level: The log level.
            event: The event instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If the event is not a `TelemetryEvent`.
        """
        if not isinstance(event, TelemetryEvent):
            raise TypeError("This logger is only for telemetry events")

        kwargs.setdefault("extra", {})["telemetry_schema"] = event
        super().log(level, event.kind, *args, **kwargs)

    def debug(self, event: TelemetryEvent, *args, **kwargs):
        """Emits a structured telemetry event at ``DEBUG`` log level.

        Args:
            event: The event instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.
        """
        self.log(logging.DEBUG, event, *args, **kwargs)

    def info(self, event: TelemetryEvent, *args, **kwargs):
        """Emits a structured telemetry event at ``INFO`` log level.

        Args:
            event: The event instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.
        """
        self.log(logging.INFO, event, *args, **kwargs)

    def warning(self, event: TelemetryEvent, *args, **kwargs):
        """Emits a structured telemetry event at ``WARNING`` log level.

        Args:
            event: The event instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.
        """
        self.log(logging.WARNING, event, *args, **kwargs)

    def error(self, event: TelemetryEvent, *args, **kwargs):
        """Emits a structured telemetry event at ``ERROR`` log level.

        Args:
            event: The event instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.
        """
        self.log(logging.ERROR, event, *args, **kwargs)

    @deprecated("Unsupported")
    def exception(self, *args) -> NoReturn:
        """Unsupported; telemetry does not handle exception logging."""
        raise NotImplementedError

    @deprecated("Unsupported")
    def critical(self, *args) -> NoReturn:
        """Unsupported; ``CRITICAL`` level is reserved for telemetry silencing."""
        raise NotImplementedError

    @deprecated("Unsupported")
    def fatal(self, *args) -> NoReturn:
        """Unsupported; ``CRITICAL`` level is reserved for telemetry silencing."""
        raise NotImplementedError
