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

from tbp.monty.telemetry.factories import _TelemetryLoggerFactory
from tbp.monty.telemetry.publishers import TelemetryPublisher


def getTelemeter(name: str) -> TelemetryPublisher:  # noqa: N802
    """Returns a telemetry logger with the specified name.

    This method is essentially a wrapper for `logger.getLogger`. It prefixes the logger
    name with ``telemetry.`` if not present, and routes logger creation through
    `_TelemetryLoggerFactory` to return a `TelemetryPublisher` instance.

    All calls to this function with a given name return the same logger instance.

    Example::

        telemeter = telemetry.getTelemeter(__name__)
        telemeter.info(TelemetryEvent(...))

    Returns:
        The logger instance of the telemetry publisher.

    Raises:
        TypeError: If the logger factory fails to return a telemetry publisher.
    """
    if not name.startswith("telemetry."):
        name = f"telemetry.{name}"

    _TelemetryLoggerFactory.setup()
    telemeter = logging.getLogger(name)

    if not isinstance(telemeter, TelemetryPublisher):
        raise TypeError(
            f"Logger factory failed to return a telemetry publisher for '{name}'"
        )

    return telemeter
