# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Telemetry log levels
NOTSET = 0
DEBUG = 10
INFO = 20
TRACE = 25


def getTelemeter(*args, **kwargs):  # noqa: N802 - lowercase
    """Alias function for the `TelemetryPublisher` constructor.

    Returns:
        The publisher instance.
    """
    # Lazy import to avoid circular dependency
    from tbp.monty.frameworks.telemetry.publishers import (  # noqa: PLC0415
        TelemetryPublisher,
    )

    return TelemetryPublisher(*args, **kwargs)
