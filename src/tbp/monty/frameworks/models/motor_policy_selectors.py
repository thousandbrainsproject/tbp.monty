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

from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.states import GoalState

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.motor_system import MotorSystem


__all__ = [
    "MotorPolicySelector",
    "SinglePolicySelector",
]


class MotorPolicySelector(Protocol):

    def pre_episode(self, motor_system: MotorSystem) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        pass

    def __call__(self, goals: list[GoalState]) -> tuple[MotorPolicy, GoalState | None]:
        pass


class SinglePolicySelector(MotorPolicySelector):
    def __init__(self, policy: MotorPolicy):
        self._policy = policy

    def pre_episode(self, motor_system: MotorSystem) -> None:
        self._policy.pre_episode(motor_system)

    def state_dict(self) -> dict[str, Any]:
        return self._policy.state_dict()

    def __call__(self, goals: list[GoalState]) -> tuple[MotorPolicy, GoalState | None]:
        if goals:
            sorted_goals = sorted(goals, key=lambda x: x.confidence, reverse=True)
            return self._policy, sorted_goals[0]
        return self._policy, None
