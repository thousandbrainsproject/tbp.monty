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

from pydantic import BaseModel, ConfigDict, Field

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.monty_base import MontyBase


class TelemetrySchema(BaseModel):
    """Base class for all telemetry schemas.

    Subclasses override ``KIND`` to identify their schema type and add fields for their
    payload. All instances carry universal context baggage (emitter, timestamp, episode,
    step, mode).
    """

    KIND: Final[str]
    """Schema kind, e.g. ``post_episode``.
    Used by `logging` for routing and as the log message in text sinks."""

    VERSION: Final[int] = 1
    """Incremented on backwards-incompatible field changes."""

    timestamp: float = Field(default_factory=time.time, kw_only=True)
    """Unix time in seconds when the schema was constructed."""

    emitter: str
    """Name of the emitting module, e.g. ``self.__class__.__name__``."""

    mode: ExperimentMode
    """Current experiment mode."""

    episode: int
    """Current episode number."""

    step: int
    """Current episode step number."""

    model_config = ConfigDict(extra="allow")
    """Allows adding instance variables to Pydantic class"""

    def populate_from_model(self, model: MontyBase):
        """Populates applicable `TelemetrySchema` values from a `MontyBase` instance."""
        self.mode = model.experiment_mode
        # TODO telemetry: self.episode = logger_args[f"{mode}_episodes"]?
        self.step = model.episode_steps


class TelemetryEvent(TelemetrySchema):
    """Base class for all telemetry events.

    Event carry instantaneous data changes, or commands like `TelemetryStopEvent`.
    """

    pass


class TelemetrySnapshot(TelemetrySchema):
    """Base class for all telemetry snapshots.

    The difference from events is that snapshots are always of level ``telemetry.TRACE``
    and contain binary or blob data.
    """

    pass


class BlankTelemetryEvent(TelemetryEvent):
    """TelemetryEvent with all fields defaulted, except ``emitter``.

    Used as a base for internal sentinel schemas that carry no meaningful payload.
    """

    mode: ExperimentMode = ExperimentMode.EVAL
    episode: int = -1
    step: int = -1


class TelemetryStopEvent(BlankTelemetryEvent):
    """Sentinel object to shut down telemetry consumer threads."""

    KIND: Final[str] = "telemetry_stop"
