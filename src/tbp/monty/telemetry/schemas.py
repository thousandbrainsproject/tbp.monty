# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated


class TelemetrySchema(BaseModel):
    """Base model class for all telemetry schemas.

    Subclasses add fields for their payload.
    """

    VERSION: Final[int] = 1
    """Schema version number, incremented on backwards-incompatible field changes.
    For use by a model validator or discriminated union."""

    kind: Annotated[str, Field(validate_default=True)] = ""
    """Schema identifier used for event filtering by subscribed handlers.
    It is also used as the log message by loggers associated with telemetry.
    If empty, defaults to schema class name."""

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value):
        return value or cls.__name__  # schema class name fallback


class TelemetryEvent(TelemetrySchema):
    """Base model class for telemetry events; carries instantaneous data changes."""

    model_config = ConfigDict(extra="allow")
    """Allows adding extra attributes to the model."""
