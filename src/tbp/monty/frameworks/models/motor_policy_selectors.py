# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    JumpToGoal,
    MotorPolicy,
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.motor_system import MotorSystem
    from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal


__all__ = [
    "MotorPolicySelector",
    "SinglePolicySelector",
    "highest_confidence_goal",
]


def highest_confidence_goal(goals: list[Goal]) -> Goal:
    """Return the goal with the highest confidence.

    If there are multiple goals with the same confidence, returns the first one.

    Args:
        goals: A list of goals. Must be non-empty.

    Returns:
        The goal with the highest confidence.

    """
    return sorted(goals, key=lambda x: x.confidence, reverse=True)[0]


@dataclass
class MotorPolicySelectorParams:
    ctx: RuntimeContext
    observations: Observations
    state: MotorSystemState
    percept: Message
    goals: list[Goal]


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
    ) -> MotorPolicyResult:
        """Return a motor policy result containing the next actions to take.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            state: The current state of the motor system.
            percept: The percept from (as of this writing) the first sensor module.
            goals: The list of goals to consider.

        Returns:
            A MotorPolicyResult that contains the actions to take.
        """
        ...


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
        goal = highest_confidence_goal(goals) if goals else None
        self._selected_goals.append(goal)
        result = self._policy(ctx, observations, state, percept, goal)
        if result is None:
            return MotorPolicyResult([])
        return result


class DistantPolicySelector(MotorPolicySelector):
    def __init__(
        self,
        jump_to_goal: JumpToGoal,
        look_at_goal: LookAtGoal,
        default: MotorPolicy,
    ):
        # policies
        self._jump_to_goal = jump_to_goal
        self._look_at_goal = look_at_goal
        self._default = default

        # state
        self._is_jumping = False

        # telemetry
        self._selected_policies: list[MotorPolicy] = []
        self._selected_goals: list[Goal | None] = []

    def pre_episode(self, motor_system: MotorSystem) -> None:
        self._jump_to_goal.pre_episode(motor_system)
        self._look_at_goal.pre_episode(motor_system)
        self._default.pre_episode(motor_system)

        self._is_jumping = False
        self._selected_policies = []
        self._selected_goals = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "jump_to_goal": self._jump_to_goal.state_dict(),
            "look_at_goal": self._look_at_goal.state_dict(),
            "default": self._default.state_dict(),
        }

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult:
        gsg_goals = [g for g in goals if g.sender_type == "GSG"]
        # Handle possibly undoing a jump or jumping to a new LM GSG goal.
        if self._is_jumping:
            goal = highest_confidence_goal(gsg_goals) if gsg_goals else None
            result = self._jump_to_goal(
                ctx,
                observations,
                state,
                percept,
                goal,
            )
            self._is_jumping = result.status == PolicyStatus.IN_PROGRESS
            if result.actions:
                self._update_telemetry(policy=self._jump_to_goal, goal=goal)
                return result

        # Handle jumping to an LM GSG's goal.
        if gsg_goals:
            goal = highest_confidence_goal(gsg_goals)
            result = self._jump_to_goal(
                ctx,
                observations,
                state,
                percept,
                goal,
            )

            self._is_jumping = result.status == PolicyStatus.IN_PROGRESS
            self._update_telemetry(policy=self._jump_to_goal, goal=goal)
            return result

        # Handle looking at an SM's goal.
        sm_goals = [g for g in goals if g.sender_type == "SM"]
        if sm_goals:
            goal = highest_confidence_goal(sm_goals)
            result = self._look_at_goal(
                ctx,
                observations,
                state,
                percept,
                goal,
            )
            self._is_jumping = False
            self._update_telemetry(policy=self._look_at_goal, goal=goal)
            return result

        # Fall back to the default policy.
        result = self._default(
            ctx,
            observations,
            state,
            percept,
            None,
        )
        self._is_jumping = False
        self._update_telemetry(policy=self._default, goal=None)
        return result

    def _update_telemetry(
        self,
        policy: MotorPolicy,
        goal: Goal | None,
    ) -> None:
        self._selected_policies.append(policy)
        self._selected_goals.append(goal)


# class SurfacePolicySelector(MotorPolicySelector):
#     def __init__(self, jump_to_goal: JumpToGoal, default: MotorPolicy):
#         self._jump_to_goal = jump_to_goal
#         self._default = default
