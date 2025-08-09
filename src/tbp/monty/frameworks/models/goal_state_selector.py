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
        """Select the best goal state from a list of goal states.

        Args:
            goal_states: An iterable containing `GoalState` (or `None`) objects.

        Returns:
            The goal state with the highest confidence value or `None` if no
            valid goal states were supplied.
        """
        # Remove "None" goal states and goal states with `use_state` equal to `False`.
        goal_states = clean_states(goal_states, unusable_ok=False)
        if len(goal_states) == 0:
            return None

        # Sort goal states by confidence
        goal_states = sort_states_by_confidence(goal_states, reverse=True)
        return goal_states[0]


def flatten(items: Iterable) -> list:
    """Recursively flatten a possibly nested iterable.

    Note that strings, while iterable, are not treated as such by this function. A
    string will be appended to the output list as-is rather than havings its
    individual characters appended.

    Args:
        items: A possibly nested iterable.

    Returns:
        list: A flat list containing all items from the input iterable.
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


def clean_states(
    states: Iterable[State | None | Iterable[State | None]],
    unusable_ok: bool = False,
) -> list[State | None]:
    """Create a flat list of `State` objects witout `None` values.

    Args:
        states: A possibly nested iterable of `State` (or `None`) objects.
        unusable_ok: Whether to include states with a `False` `use_state` attribute in
            the returned list. Defaults to `False`.

    Returns:
        A flat list of `State` objects.
         - May include states with a `False` `use_state` attribute if `unusable_ok`
           is `True`.
    """
    # Flatten states since they may be nested.
    flat = flatten(states)

    # Remove "None" states.
    clean = [s for s in flat if s is not None]

    # Remove unusable goal states if not allowed.
    if not unusable_ok:
        clean = [s for s in clean if s.use_state]

    return clean


def sort_states_by_confidence(
    states: Iterable[State],
    reverse: bool = False,
) -> list[State]:
    """Sort states by confidence.

    Args:
        states: State objects to sort.
        reverse: Whether to sort in reverse order. Defaults to `False`. Set to `True`
            to have states with higher confidence values first.

    Returns:
        Input states sorted by confidence.
    """
    confidence = np.array([s.confidence for s in states], dtype=float)

    # Handle None/np.nan confidence values (set them to -np.inf).
    confidence[np.isnan(confidence)] = -np.inf

    # Find sorting indices, possibly reversing them.
    sorting_inds = np.argsort(confidence)
    if reverse:
        sorting_inds = sorting_inds[::-1]

    return [states[i] for i in sorting_inds]
