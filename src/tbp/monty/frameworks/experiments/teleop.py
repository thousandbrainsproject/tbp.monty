# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

from typing_extensions import Self

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations


class Teleop(Protocol):
    """Teleoperation protocol for visualization and control."""

    def __call__(
        self: Self,
        ctx: RuntimeContext,
        monty: Monty,
        supervised_lm_ids: list[str],
        step: int,
        observations: Observations,
        actions: list[Action],
    ) -> list[Action]:
        """Teleoperate the Monty model.

        Args:
            ctx: The runtime context.
            monty: The Monty model.
            supervised_lm_ids: The list of supervised learning module IDs.
            step: The current step.
            observations: The observations.
            actions: The actions to take.

        Returns:
            The actions to take.
        """
        ...

    def close(self) -> None:
        """Close the teleoperation."""
        ...


class TeleopNoOp(Teleop):
    """Teleoperation no-op implementation."""

    def __call__(
        self: Self,
        ctx: RuntimeContext,  # noqa: ARG002
        monty: Monty,  # noqa: ARG002
        supervised_lm_ids: list[str],  # noqa: ARG002
        step: int,  # noqa: ARG002
        observations: Observations,  # noqa: ARG002
        actions: list[Action],
    ) -> list[Action]:
        return actions

    def close(self) -> None:
        pass
