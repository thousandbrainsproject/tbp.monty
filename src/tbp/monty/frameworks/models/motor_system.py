# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policy_selectors import MotorPolicySelector
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.models.states import GoalState, State

__all__ = ["MotorSystem"]


@dataclass
class SurfacePolicyActionDetailsTelemetry:
    pc_heading: list[Literal["min", "max", "no", "jump"] | None]
    avoidance_heading: list[bool | None]
    z_defined_pc: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]] | None]


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(self, policy_selector: MotorPolicySelector) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy_selector: The motor policy selector to use.
        """
        self._policy_selector = policy_selector
        # For each step, we store the actions produced by the policy and the current
        # motor system state as a (actions, state) tuple.
        self._action_sequence: list[tuple[list[Action], dict[AgentID, Any] | None]] = []

        # TODO: When the motor system is encapsulated within Monty, then motor_only_step
        #       attribute should be moved to Monty itself instead.
        self.motor_only_step = False

        # TODO: Get rid of this once we have another path for telemetry.
        self._telemetry_surface_action_details = SurfacePolicyActionDetailsTelemetry(
            pc_heading=[],
            avoidance_heading=[],
            z_defined_pc=[],
        )

    @property
    def action_sequence(self) -> list[tuple[list[Action], dict[AgentID, Any] | None]]:
        return self._action_sequence

    def pre_episode(self) -> None:
        """Pre episode hook."""
        # TODO: Passing self to policy pre_episode is a hack. What we should be
        # doing is using a positioning procedure for surface agents instead.
        # We only do this so that SurfacePolicy and its descendants can set
        # motor_only_step to True.
        # Undoing this hack should probably happen when motor_only_step is moved
        # to Monty itself.
        self._policy_selector.pre_episode(self)
        self._action_sequence = []
        self._telemetry_surface_action_details = SurfacePolicyActionDetailsTelemetry(
            pc_heading=[],
            avoidance_heading=[],
            z_defined_pc=[],
        )

    def state_dict(self) -> dict[str, Any]:
        return self._policy_selector.state_dict()

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
        percept: State,
        goals: list[GoalState],
    ) -> list[Action]:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.
            percept: The percept from (as of this writing) the first sensor
                module.
            goals: The goals to consider.

        Returns:
            The action to take.
        """
        motor_system_state = MotorSystemState(proprioceptive_state)
        policy, goal = self._policy_selector(goals)
        policy_result = policy(ctx, observations, motor_system_state, percept, goal)
        self.motor_only_step = policy_result.motor_only_step

        state_copy = motor_system_state.convert_motor_state()
        self._action_sequence.append((policy_result.actions, state_copy))

        telemetry = policy_result.telemetry
        if telemetry is not None:
            self._telemetry_surface_action_details.pc_heading.append(
                telemetry.pc_heading
            )
            self._telemetry_surface_action_details.avoidance_heading.append(
                telemetry.avoidance_heading
            )
            self._telemetry_surface_action_details.z_defined_pc.append(
                telemetry.z_defined_pc
            )

        return policy_result.actions
