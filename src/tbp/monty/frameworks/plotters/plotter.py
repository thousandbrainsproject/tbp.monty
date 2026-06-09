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

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations


class Plotter(Protocol):
    interactive: bool

    def initialize(self, model: Monty) -> None:
        """Resolve the displayed learning module and build the figure and widgets.

        Called once per episode.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
        """
        ...

    def update(self, observations: Observations, step: int) -> None:
        """Draw the current state.

        Never blocks for an action (interactive blocking lives in `override_action`);
        a non-interactive plotter may pause or halt here according to its speed slider.

        Args:
            observations: The observations from the most recent step.
            step: The index of the current step within the episode.
        """
        ...

    def awaits_choice(self, proposed: list[Action]) -> bool:
        """Whether the user should choose this step's action (interactive only).

        Args:
            proposed: The actions the model computed for this step.

        Returns:
            True when this step is a user choice point.
        """
        ...

    def override_action(
        self, ctx: RuntimeContext, proposed: list[Action]
    ) -> list[Action]:
        """Return the user's chosen action (interactive only).

        Block until a button is clicked, then return the user's chosen action.

        Args:
            ctx: The runtime context supplying the random state.
            proposed: The actions the model computed for this step, offered as a
                "jump" choice when they are a hypothesis-testing jump.

        Returns:
            The actions to execute next, built from the user's button choice.

        Raises:
            StopIteration: When the user clicks "End episode".
        """
        ...

    def close(self) -> None: ...
