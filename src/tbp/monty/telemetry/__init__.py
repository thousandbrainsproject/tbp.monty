# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Structured telemetry framework built upon the `logging` module.

Provides a structured telemetry module that emits inline telemetry events routed through
standard Python `logging` mechanics. Telemetry schemas and events are passed via the
``extra`` parameter of `logging.Logger.log` and stored within the internal ``__dict__``
of `logging.LogRecord` objects.

The telemetry level must be configured via the experiment config YAML. Easiest is adding
"  - /telemetry: info" under "defaults:". Available configs are "info", "debug", "none".

The global level is defined via the ``telemetry.tbp.monty`` logger. It can be overridden
on a per-module basis.

Config example::

    experiment:
      config:
        telemetry:
          loggers:
            # Global level
            telemetry.tbp.monty:
              level: CRITICAL
            # Module-specific level
            telemetry.tbp.monty.frameworks.models.graph_matching:
              level: INFO

Usage example::

    from tbp.monty import telemetry
    from tbp.monty.telemetry.schemas import TelemetryEvent

    telemeter = telemetry.getTelemeter(__name__)
    telemeter.info(TelemetryEvent(kind="CustomEvent", your_key="your_value", ...))
    telemeter.debug(TelemetryEvent(kind="DebugEvent", ...))
"""

from tbp.monty.telemetry.telemeter import getTelemeter

__all__ = ["getTelemeter"]
