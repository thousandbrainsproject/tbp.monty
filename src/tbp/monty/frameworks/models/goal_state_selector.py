# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from tbp.monty.frameworks.models.states import GoalState, State

logger = logging.getLogger(__name__)


class GoalStateSelector:
    """Monty component used to select a single goal state from many."""

    def select(
        self,
        goal_states: Iterable[GoalState | None | Iterable[GoalState | None]],
    ) -> GoalState | None:
        """Select the best goal state from zero or more goal states.

        Since goal-state-generators can return `None`, a single goal-state, or an
        iterable of goal states, pooling the output of several GSGs can result in
        messy combinations of allowed returned values.

        This function first sanitizes the input goal states by putting all valid
        goal states into a single flat list.

        Args:
            goal_states: A possibly nested iterable of `GoalState` (or `None`) objects.

        Returns:
            The goal state with the highest confidence value or `None` if no
            valid goal states were supplied.
        """
        # Get all items into one flat list.
        goal_states = flatten(goal_states)

        # Drop `None` goal states.
        goal_states = [s for s in goal_states if s is not None]

        # Drop states with `use_state` equal to `False`.
        goal_states = [s for s in goal_states if s.use_state]

        # Quick out if no goal states remain.
        if len(goal_states) == 0:
            return None

        # Sort states by confidence value.
        goal_states = sort_states_by_confidence(goal_states, reverse=True)

        # Select the goal state with the highest confidence value.
        highest_confidence_goal_state = goal_states[0]

        # TODO: Figure out how to prevent returning the same location.
        # TODO: Log the selected goal state.

        return highest_confidence_goal_state


def flatten(items: Iterable) -> list:
    """Recursively flatten a possibly nested iterable.

    Note that strings, while iterable, are not treated as such by this function. A
    string will be appended to the output list as-is rather than havings its
    individual characters appended.

    Args:
        items: A possibly nested iterable.

    Returns:
        list: A flat list containing all items from the input iterable.

    Example:
        >>> flatten([1, 2, [3, "four"], [5, (6, 7)]])
        [1, 2, 3, "four", 5, 6, 7]

    TODO: Figure out where to put this function. It's used by `GoalStateSelector`, but
    it's functionality isn't specific to `State` or `GoalState` objects.

    """
    if isinstance(items, str):
        return [items]

    flat = []
    for elt in items:
        if isinstance(elt, Iterable) and not isinstance(elt, str):
            flat.extend(flatten(elt))
        else:
            flat.append(elt)

    return flat


def sort_states_by_confidence(
    states: Iterable[State],
    reverse: bool = False,
) -> list[State]:
    """Sort states according to their confidence values.

    Args:
        states: State objects to sort.
        reverse: Whether to sort in reverse order. Defaults to `False`. Set to `True`
            to have states with highest confidence values first.

    Returns:
        Input states sorted by confidence value.

    TODO: Maybe find another home for this function. It's used by `GoalStateSelector`,
    but it works with regular `State` objects, not just `GoalState` objects, so maybe
    it belongs in a `states.py` or a utility module.
    """
    confidence = np.array([s.confidence for s in states], dtype=float)

    # Handle None/np.nan confidence values by setting them to -np.inf.
    confidence[np.isnan(confidence)] = -np.inf

    # Find sorting indices, possibly reversing them.
    sorting_inds = np.argsort(confidence)
    if reverse:
        sorting_inds = sorting_inds[::-1]

    return [states[i] for i in sorting_inds]
