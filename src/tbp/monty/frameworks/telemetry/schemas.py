# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import sys
import time
from typing import Final, Mapping

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated


class TelemetrySchema(BaseModel):
    """Base model class for all telemetry schemas.

    Subclasses add fields for their payload.
    """

    VERSION: Final[int] = 1
    """Schema version number, incremented on backwards-incompatible field changes.
    For use by a model validator or discriminated union."""

    kind: Annotated[str, Field(validate_default=True)] = ""
    """Schema identifier used by telemetry loggers as the log message in text sinks.
    It can also be used for event filtering by subscribed handlers.
    If empty, defaults to schema class name."""

    timestamp: float = Field(default_factory=time.time, kw_only=True)
    """Unix time in seconds when the schema was instantiated."""

    origin: str = Field(default_factory=lambda: sys._getframe(2).f_globals["__name__"])
    """Name of module where schema was instantiated. Auto-populated, but overridable."""

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value):
        return value or cls.__name__  # schema class name fallback


class TelemetryEvent(TelemetrySchema):
    """Base model class for telemetry events; carries instantaneous data changes."""

    values: Mapping = {}
    """Generic dict for miscellaneous values to pass along."""
