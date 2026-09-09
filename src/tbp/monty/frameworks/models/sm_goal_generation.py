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

if TYPE_CHECKING:
    from tbp.monty.cmp import Goal, Message
    from tbp.monty.frameworks.models.motor_system_state import SensorState
    from tbp.monty.memento import Memento


class SMGoalGenerator(Protocol):
    """A sensor module's goal generator: its counterpart of a learning module's GSG.

    Called by its sensor module once per step with the percept and the
    sensor's pose; returns the goals the module proposes to the motor system
    this step (empty when it proposes none). What it records about the
    episode is exposed through state_dict for telemetry.
    """

    def __call__(
        self,
        percept: Message,
        sensor_state: SensorState,
        motor_only_step: bool = False,
    ) -> list[Goal]: ...

    def reset(self) -> None:
        """Forget any per-episode state."""

    def state_dict(self) -> Memento: ...


class NoopSMGoalGenerator(SMGoalGenerator):
    """Never proposes anything."""

    def __call__(
        self,
        percept: Message,  # noqa: ARG002
        sensor_state: SensorState,  # noqa: ARG002
        motor_only_step: bool = False,  # noqa: ARG002
    ) -> list[Goal]:
        return []

    def reset(self) -> None:
        pass

    def state_dict(self) -> Memento:
        return {}
