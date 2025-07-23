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

from tbp.monty.frameworks.models.states import GoalState

logger = logging.getLogger(__name__)


class GoalStateSelector:
    """Select the best goal state from a list of goal states."""

    def select(self, goal_states: list[GoalState]) -> GoalState | None:
        """Select the best goal state from a list of goal states.

        Args:
            goal_states: A list of goal states.

        Returns:
            The best goal state.
        """
        # Remove "None" goal states
        goal_states = [gs for gs in goal_states if gs is not None and gs.use_state]
        if not goal_states:
            return None
        else:
            # Sort goal states by confidence
            goal_states = sorted(goal_states, key=lambda x: x.confidence, reverse=True)
            return goal_states[0]
