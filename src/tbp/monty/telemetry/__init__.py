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

import logging
from typing import ClassVar

from tbp.monty.telemetry.publishers import TelemetryPublisher as _TelemetryPublisher
from tbp.monty.telemetry.telemeter import getTelemeter

__all__ = ["getTelemeter"]


class _TelemetryLoggerFactory(logging.Logger):
    """Routes ``telemetry.*`` names to ``TelemetryPublisher``-typed loggers.

    ``logging`` notes that "Loggers should _NEVER_ be instantiated directly, but always
    through the module-level function ``logging.getLogger(name)``."

    Installed as ``Logger.manager.loggerClass``, this class is invoked by
    ``Manager.getLogger`` to construct loggers it hasn't seen: ``rv = (self.loggerClass
    or _loggerClass)(name)``. Names not prefixed with ``telemetry.*`` are passed through
    to whatever logger class the process would otherwise have used.

    A factory is needed here because ``TelemetryPublisher`` subclasses
    ``logging.Logger`` and so, per ``logging``'s rules, must never be instantiated
    directly. This factory class is itself never instantiated either: its ``__init__``
    never runs, as its ``__new__`` always returns another object (hence the ``# type:
    ignore[misc]``).

    It is installed on the manager rather than globally via ``logging.setLoggerClass``
    so it can't be picked up as a base class by libraries that do ``class
    LibLogger(logging.getLoggerClass()): ...``, which would block their subclass from
    being instantiated. It is decoupled from ``logging.getLoggerClass()``, which avoids
    the possibility of accidental inheritance, while still being checked first by
    ``Manager.getLogger``, so telemetry names are still always caught.

    This placement also ensures thread-safety: ``Manager.getLogger`` calls the factory
    while holding the logging lock, so dispatch happens atomically with the logger
    dictionary, unlike swapping the global logger class around a ``getLogger()`` call,
    which would race with other threads.

    The non-telemetry branch resolves the fallback class on every call (so a class
    installed later is honored), falling back to ``_manager_logger_class`` to preserve
    any manager-level class set before this one.
    """

    _manager_logger_class: ClassVar = logging.Logger.manager.loggerClass

    def __new__(cls, name, *args, **kwargs) -> logging.Logger:  # type: ignore[misc]
        # TODO telemetry: add "snapshots." here in the future
        if name.startswith("telemetry."):
            return _TelemetryPublisher(name, *args, **kwargs)

        logger_class = cls._manager_logger_class or logging.getLoggerClass()
        return logger_class(name, *args, **kwargs)


logging.Logger.manager.setLoggerClass(_TelemetryLoggerFactory)
