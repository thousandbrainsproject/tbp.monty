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
from typing import Final

from pydantic import BaseModel, Field

from tbp.monty.frameworks.experiments.mode import ExperimentMode


class TelemetryEvent(BaseModel):
    """Base class for all telemetry snapshot events.

    Subclasses override SCHEMA_ID to identify their event type and add fields for their
    payload. All instances carry universal context baggage (emitter, timestamp, episode,
    step, mode).
    """

    SCHEMA_ID: Final[str]
    """Event type identifier, e.g. ``post_episode``.
    Used by TelemetryBroker for routing and as the log message in text sinks."""

    SCHEMA_VERSION: Final[int] = 1
    """Incremented on backwards-incompatible field changes."""

    timestamp: float = Field(default_factory=time.time, kw_only=True)
    """Unix time in seconds when the event was captured."""

    emitter: str
    """Name of the emitting module, e.g. ``self.__class__.__name__``."""

    mode: ExperimentMode
    """Current experiment mode."""

    episode: int
    """Current episode number."""

    step: int
    """Number of overall steps, including those where no LM update was performed."""


class _BlankTelemetryEvent(TelemetryEvent):
    """TelemetryEvent with all fields defaulted.

    Used as a base for internal sentinel events that carry no meaningful payload.
    """

    emitter: str = ""
    mode: ExperimentMode = ExperimentMode.EVAL
    episode: int = -1
    step: int = -1


class TelemetryStopEvent(_BlankTelemetryEvent):
    """Sentinel object to shut down telemetry consumer threads."""

    SCHEMA_ID: Final[str] = "telemetry_stop"
