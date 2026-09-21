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

from typing_extensions import deprecated, override

from tbp.monty.telemetry.schemas import TelemetryEvent

# ruff: noqa: DOC502


class TelemetryPublisher(logging.Logger):
    """Structured telemetry publisher.

    Subclasses `logging.Logger` and emits `TelemetrySchema` as structured `LogRecord`
    instances routed through the logging pipeline to telemetry handlers.

    The `TelemetryEvent` is passed as the log message, so it is available to handlers as
    ``record.msg``; ``record.getMessage()`` returns ``event.kind``.

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

    def _log_event(self, level: int, msg: object, *args, **kwargs):
        """Catch-all internal method for emitting telemetry events.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        if not isinstance(msg, TelemetryEvent):
            raise TypeError("This method is only for telemetry events")

        # skip 2 frames to get to the caller
        kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 2

        super().log(level, msg, *args, **kwargs)

    @override
    def log(self, level: int, msg: object, *args, **kwargs):
        """Emits a telemetry event at the specified log level.

        Args:
            level: The log level.
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(level, msg, *args, **kwargs)

    # Type-hinted equivalent of `log` for convenience
    def emit(self, level: int, event: TelemetryEvent, *args, **kwargs):
        """Emits a telemetry event at the specified log level.

        Args:
            level: The log level.
            event: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(level, event, *args, **kwargs)

    @override
    def debug(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``DEBUG`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.DEBUG, msg, *args, **kwargs)

    @override
    def info(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``INFO`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.INFO, msg, *args, **kwargs)

    @override
    def warning(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``WARNING`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.WARNING, msg, *args, **kwargs)

    @deprecated("Deprecated since Python 3.3. Use `warning()` instead.")
    @override
    def warn(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``WARNING`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.WARNING, msg, *args, **kwargs)

    @override
    def error(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``ERROR`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.ERROR, msg, *args, **kwargs)

    @override
    def exception(self, msg: object, *args, exc_info=True, **kwargs):
        """Emits a telemetry event at ``ERROR`` log level with exception info attached.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            exc_info: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    @override
    def critical(self, msg: object, *args, **kwargs):
        """Emits a telemetry event at ``CRITICAL`` log level.

        Args:
            msg: The `TelemetryEvent` instance.
            *args: Passed forward to ``Logger.log`` method.
            **kwargs: Passed forward to ``Logger.log`` method.

        Raises:
            TypeError: If ``msg`` is not a `TelemetryEvent`.
        """
        self._log_event(logging.CRITICAL, msg, *args, **kwargs)

    # ``Logger.fatal`` is aliased to ``Logger.critical``, so it must be re-aliased here
    fatal = critical
