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
from typing import ClassVar

from tbp.monty.telemetry.publishers import TelemetryPublisher


class _TelemetryLoggerFactory(logging.Logger):
    """Internal factory class for use via `telemetry.getTelemeter`.

    Produces a new `TelemetryPublisher` instance if the logger name is prefixed by
    ``telemetry.``. Otherwise, produces a new instance of the default logger class.

    This factory is needed because new `Logger` objects must be instantiated from inside
    `logging` for its internal mechanics to work properly. The `setup` method calls
    `logging.setLoggerClass` to designate the factory as the `logging` module's main
    provider of new `Logger` objects.
    """

    _logger_class: ClassVar = logging.getLoggerClass()

    @classmethod
    def setup(cls):
        """Ensures this factory is designated as the main logger class."""
        logger_class = logging.getLoggerClass()
        if not issubclass(logger_class, cls):
            cls._logger_class = logger_class
            logging.setLoggerClass(cls)

    def __new__(cls, name, *args, **kwargs) -> logging.Logger:
        """Instantiates and returns a new logger; subclass varies with prefix."""
        # TODO telemetry: add "snapshots." here in the future
        if name.startswith("telemetry."):
            return TelemetryPublisher(name, *args, **kwargs)

        return cls._logger_class(name, *args, **kwargs)
