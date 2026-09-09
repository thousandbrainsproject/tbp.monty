# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import (
    ExperimentMotorPolicySelector,
    ExperimentMotorSystem,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    JumpToGoal,
    MotorPolicy,
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.memento import Memento

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal


__all__ = [
    "DistantPolicySelector",
    "MotorPolicySelector",
    "RuntimeMotorPolicySelector",
    "SinglePolicySelector",
    "highest_confidence_goal",
]


def specifies_pose(goal: Goal) -> bool:
    """Whether a goal asks for an agent pose rather than just a place to look at.

    A goal carrying pose vectors (an LM's hypothesis-testing goal, a sensor
    module's re-orienting goal) wants the agent moved into that pose; one
    without (a salience goal) only names a location to look toward.

    Args:
        goal: The goal.

    Returns:
        True when the goal's morphological features hold pose vectors.
    """
    features = goal.morphological_features
    return bool(features) and features.get("pose_vectors") is not None


def highest_confidence_goal(goals: Sequence[Goal]) -> Goal:
    """Return the goal with the highest confidence.

    If there are multiple goals with the same confidence, returns the first one.

    Args:
        goals: A sequence of goals. Must be non-empty.

    Returns:
        The goal with the highest confidence.

    """
    return sorted(goals, key=lambda x: x.confidence, reverse=True)[0]


class RuntimeMotorPolicySelector(Protocol):
    """Monty runtime interface to a Motor Policy Selector."""

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: Sequence[Goal],
    ) -> MotorPolicyResult:
        """Return a motor policy result containing the next actions to take.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            state: The current state of the motor system.
            percept: The percept from (as of this writing) the first sensor module.
            goals: The goals to consider.

        Returns:
            A MotorPolicyResult that contains the actions to take.
        """
        ...


class MotorPolicySelector(
    RuntimeMotorPolicySelector, ExperimentMotorPolicySelector, Protocol
):
    pass


class SinglePolicySelector(MotorPolicySelector):
    def __init__(self, policy: MotorPolicy):
        self._policy = policy
        # TODO: Get rid of this once we have another path for telemetry.
        self._selected_goals: list[Goal | None] = []

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        self._policy.fixme_provide_motor_system(motor_system)

    def reset(self) -> None:
        self._policy.reset()
        self._selected_goals = []

    def state_dict(self) -> Memento:
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
        goals: Sequence[Goal],
    ) -> MotorPolicyResult:
        goal = highest_confidence_goal(goals) if goals else None
        self._selected_goals.append(goal)
        return self._policy(ctx, observations, state, percept, goal)


class DistantPolicySelector(MotorPolicySelector):
    """Pick the distant agent's policy from the goals it was given.

    Goals that specify a pose (see ``specifies_pose``) are jumped to, the
    highest-confidence one first; goals that only name a location are looked
    at; with no goals at all, the default policy explores.
    """

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

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        self._jump_to_goal.fixme_provide_motor_system(motor_system)
        self._look_at_goal.fixme_provide_motor_system(motor_system)
        self._default.fixme_provide_motor_system(motor_system)

    def reset(self) -> None:
        self._jump_to_goal.reset()
        self._look_at_goal.reset()
        self._default.reset()

        self._is_jumping = False
        self._selected_policies = []
        self._selected_goals = []

    def state_dict(self) -> Memento:
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
        goals: Sequence[Goal],
    ) -> MotorPolicyResult:
        pose_goals = [g for g in goals if specifies_pose(g)]
        # Handle possibly undoing a jump or jumping to a new pose goal.
        if self._is_jumping:
            goal = highest_confidence_goal(pose_goals) if pose_goals else None
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

        # Handle jumping to a pose goal.
        if pose_goals:
            goal = highest_confidence_goal(pose_goals)
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

        # Handle looking at a location-only goal.
        look_goals = [g for g in goals if not specifies_pose(g)]
        if look_goals:
            goal = highest_confidence_goal(look_goals)
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
