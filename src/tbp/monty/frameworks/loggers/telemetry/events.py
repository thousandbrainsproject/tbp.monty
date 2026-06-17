# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

from tbp.monty.frameworks.experiments.mode import ExperimentMode


@dataclass
class TelemetryEvent:
    """Base class for all telemetry snapshot events.

    Subclasses declare a schema_id to identify their event type and add dataclass fields
    for their payload. All instances carry universal context baggage (emitter,
    timestamp, episode, step, mode).
    """

    schema_id: ClassVar[str] = ""
    """Event type identifier, e.g. `env_interface.step`.
    Used by TelemetryBroker for routing and as the log message in text sinks."""

    schema_version: ClassVar[int] = 1
    """Incremented on backwards-incompatible field changes."""

    emitter: str = ""
    """Name of the emitting module, e.g. `self.__class__.__name__`."""

    timestamp: float = field(default_factory=time.monotonic)
    episode: int = 0
    step: int = 0
    mode: ExperimentMode = ExperimentMode.EVAL


class TelemetryStopEvent(TelemetryEvent):
    """Sentinel object to shut down telemetry consumer threads."""

    pass


TELEMETRY_STOP = TelemetryStopEvent()
