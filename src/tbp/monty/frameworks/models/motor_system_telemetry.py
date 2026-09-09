# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tbp.monty.cmp import goals_to_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tbp.monty.cmp import Goal
    from tbp.monty.memento import Memento

__all__ = [
    "MotorSystemTelemetry",
    "MotorSystemTelemetryProtocol",
    "NoopMotorSystemTelemetry",
]


class MotorSystemTelemetryProtocol(Protocol):
    """What a :class:`MotorSystem` reports to its telemetry."""

    def reset(self) -> None: ...

    def goals_in(self, goals: Sequence[Goal]) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopMotorSystemTelemetry(MotorSystemTelemetryProtocol):
    """Records nothing; the default."""

    def reset(self) -> None:
        pass

    def goals_in(self, goals: Sequence[Goal]) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(goals_in=[])


class MotorSystemTelemetry(MotorSystemTelemetryProtocol):
    """Keeps the goals the motor system received each step, as ``goals_in``.

    Each step's goals are kept as columns (see
    :func:`~tbp.monty.cmp.goals_to_columns`) rather than as ``Goal`` objects.
    """

    def __init__(self) -> None:
        self._goals_in: list[dict[str, object]] = []

    def reset(self) -> None:
        self._goals_in = []

    def goals_in(self, goals: Sequence[Goal]) -> None:
        self._goals_in.append(goals_to_columns(goals))

    def state_dict(self) -> Memento:
        return dict(goals_in=list(self._goals_in))
