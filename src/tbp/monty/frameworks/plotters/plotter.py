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

from numpy.random import RandomState

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations


class Plotter(Protocol):
    interactive: bool

    def initialize(self, model: Monty) -> None:
        """Resolve the target SM/LM and build the figure/axes/widgets.

        Called once per episode.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
        """
        ...

    def update(self, observations: Observations, step: int) -> None:
        """Draw the current state.

        Never blocks for an action (interactive blocking lives in
        override_action); a non-interactive plotter may pause/halt here per its
        speed slider.

        Args:
            observations: The observations from the most recent step.
            step: The index of the current step within the episode.
        """
        ...

    def override_action(self, rng: RandomState) -> list[Action]:
        """Return the user's chosen action (interactive only).

        Block until a button is clicked, then return the user's chosen action.

        Args:
            rng: The random state used to sample the chosen action.

        Returns:
            The actions to execute next, built from the user's button choice.

        Raises:
            StopIteration: When the user clicks "End episode".
        """
        ...

    def close(self) -> None: ...
