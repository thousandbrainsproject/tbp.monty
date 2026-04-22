# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy, MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.motor_system import MotorSystem


__all__ = [
    "MotorPolicySelector",
    "SinglePolicySelector",
]


class MotorPolicySelector(Protocol):
    def pre_episode(self, motor_system: MotorSystem) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult: ...


class SinglePolicySelector(MotorPolicySelector):
    def __init__(self, policy: MotorPolicy):
        self._policy = policy
        # TODO: Get rid of this once we have another path for telemetry.
        self._selected_goals: list[Goal | None] = []

    def pre_episode(self, motor_system: MotorSystem) -> None:
        self._policy.pre_episode(motor_system)
        self._selected_goals = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self._policy.state_dict(),
            "selected_goals": self._selected_goals,
        }

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult:
        if goals:
            sorted_goals = sorted(goals, key=lambda x: x.confidence, reverse=True)
            goal = sorted_goals[0]
        else:
            goal = None
        self._selected_goals.append(goal)
        return self._policy(ctx, observations, state, percept, goal)
