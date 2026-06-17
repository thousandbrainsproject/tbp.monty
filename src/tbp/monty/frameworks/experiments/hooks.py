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

__all__ = [
    "NoOpStepHook",
    "StepHook",
]


class StepHook(Protocol):
    """Step hook protocol for customizing the step behavior."""

    def __call__(
        self: Self,
        ctx: RuntimeContext,
        monty: Monty,
        supervised_lm_ids: list[str],
        step: int,
        observations: Observations,
        actions: list[Action],
    ) -> list[Action]:
        """Execute the step hook.

        The step hook is used to customize the step behavior on behalf of the
        experiment. The hook occurs after the Monty model output actions intended for
        the environment, but before the environment is stepped.

        Having a hook at this point in the execution loop allows for visualization of
        observations, resultant actions, as well as any internal state of the Monty
        model available to the experiment. Additionally, the hook can be used to
        augment, override, modify, delete, or otherwise customize the actions intended
        for the environment.

        Args:
            ctx: The runtime context.
            monty: The Monty model.
            supervised_lm_ids: The list of supervised learning module IDs.
            step: The current step.
            observations: The observations provided to the Monty model.
            actions: The actions output by the Monty model intended for the environment.

        Returns:
            The actions to take by the environment.
        """
        ...

    def close(self) -> None:
        """Close the step hook."""
        ...


class NoOpStepHook(StepHook):
    """Step hook no-op implementation."""

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
