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

from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import Annotated, Self

from tbp.monty.frameworks.experiments.mode import ExperimentMode


class TelemetrySchema(BaseModel):
    """Base model class for all telemetry schemas.

    Subclasses override ``KIND`` to identify their schema type and add fields for their
    payload.
    """

    VERSION: Final[int] = 1
    """Incremented on backwards-incompatible field changes."""

    timestamp: float = Field(default_factory=time.time, kw_only=True)
    """Unix time in seconds when the schema was constructed."""

    emitter: Annotated[
        str | object,
        BeforeValidator(
            lambda x: x
            if isinstance(x, str)
            else f"{x.__class__.__module__}.{x.__class__.__name__}"
        ),
    ]
    """Module and name of the emitting class; object values auto-converted to string."""

    @property
    def kind(self) -> str:
        """Schema identifier used by `logging` as the log message in text sinks."""
        return self.__class__.__name__

    def shallow_copy(self, emitter: str | object) -> Self:
        """Shallow-copies a schema, with explicit emitter override.

        Returns:
            The new schema instance.
        """
        return self.model_copy(update={"emitter": emitter})


class TelemetryEvent(TelemetrySchema):
    """Base model class for telemetry events.

    Carries instantaneous data changes.
    """

    pass


class TelemetrySnapshot(TelemetrySchema):
    """Base model class for telemetry snapshots.

    The difference from events is that snapshots are always of level ``telemetry.TRACE``
    and contain binary or blob data.
    """

    pass


class EpisodeTelemetryMixin(BaseModel):
    """Base model mixin for telemetry schemas with fields for episode state.

    All instances carry universal context baggage (emitter, timestamp, episode, step,
    mode).
    """

    mode: ExperimentMode
    """Current experiment mode."""

    episode: int
    """Current episode number."""

    step: int
    """Current episode step number."""

    def __init_subclass__(cls, **kwargs):
        """Ensures the mixin is used only with `TelemetrySchema` subclasses.

        Raises:
            TypeError: If the mixin is used with a non-compatible class.
        """
        super().__init_subclass__(**kwargs)
        if not any(issubclass(b, TelemetrySchema) for b in cls.__bases__):
            raise TypeError(
                "Mixin requires a subclass of "
                f"{TelemetrySchema.__name__}, got {cls.__bases__}"
            )


class EpisodeTelemetryEvent(EpisodeTelemetryMixin, TelemetryEvent):
    """Base model class for episode-related telemetry events."""

    pass


class EpisodeTelemetrySnapshot(EpisodeTelemetryMixin, TelemetrySnapshot):
    """Base model class for episode-related telemetry snapshots."""

    pass
