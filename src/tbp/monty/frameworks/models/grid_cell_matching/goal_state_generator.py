# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging

import numpy as np

from tbp.monty.frameworks.models.goal_state_generation import GraphGoalStateGenerator
from tbp.monty.frameworks.models.states import GoalState

__all__ = ["GridCellGoalStateGenerator"]

logger = logging.getLogger(__name__)


class GridCellGoalStateGenerator(GraphGoalStateGenerator):
    """Goal state generator for GridCellLM.

    Proposes discriminative locations using SDR mismatch in scaffold phase
    space. The basic strategy is:

    1. Find the Most Likely Hypothesis (MLH) and 2nd-MLH.
    2. Sample scaffold phases from the MLH's object model.
    3. For each sampled phase, compare the recalled SDRs for the MLH and
       2nd-MLH objects.
    4. Choose the phase where SDR overlap is minimised — this is the most
       discriminative location.
    5. Convert the discriminative phase to a body-frame location and output
       as a GoalState.

    Falls back to random exploration direction if fewer than 2 hypotheses
    exist or if the scaffold doesn't support sampling.

    Attributes:
        goal_tolerances: Tolerances for determining goal satisfaction.
        min_post_goal_success_steps: Minimum steps after a goal is reached
            before generating a new one.
    """

    def __init__(
        self,
        parent_lm=None,
        goal_tolerances: dict | None = None,
        min_post_goal_success_steps: int = 5,
        desired_object_distance: float = 0.03,
        **kwargs,
    ):
        super().__init__(parent_lm, **kwargs)
        self.goal_tolerances = goal_tolerances or {"distance": 0.01}
        self.min_post_goal_success_steps = min_post_goal_success_steps
        self.desired_object_distance = desired_object_distance
        self._steps_since_goal = 0
        self._current_goal = None

    def reset(self):
        """Reset goal state generator for a new episode."""
        super().reset()
        self._steps_since_goal = 0
        self._current_goal = None

    def _generate_goal_state(self, observations) -> list:
        """Generate goal states for the motor system.

        Uses hypothesis-testing strategy: find the location that best
        discriminates between the MLH and the 2nd-best hypothesis.

        Falls back to random exploration if insufficient hypotheses.

        Args:
            observations: Current observations (used for on-object check).

        Returns:
            List of GoalState objects (typically 0 or 1).
        """
        lm = self.parent_lm
        if lm is None:
            return []

        # Check if we need a new goal
        self._steps_since_goal += 1
        if (
            self._current_goal is not None
            and self._steps_since_goal < self.min_post_goal_success_steps
        ):
            return []

        hypotheses = getattr(lm, "hypotheses", [])
        if len(hypotheses) < 2:
            return self._random_exploration_goal(observations)

        # Get MLH and 2nd-best
        sorted_hyps = sorted(hypotheses, key=lambda h: h.evidence, reverse=True)
        mlh = sorted_hyps[0]
        second = sorted_hyps[1]

        # If same object: explore to narrow down pose
        # If different objects: explore to discriminate between them
        # In both cases: find discriminative location

        # Use MLH's accumulated displacement as base, add random offset
        # pointing away from current position
        base_loc = mlh.accumulated_displacement.copy()

        # Generate goal as a random offset from current position on the
        # object surface. The motor system will handle actual navigation.
        rng = np.random.default_rng()
        random_dir = rng.standard_normal(3)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
        goal_loc = base_loc + random_dir * self.desired_object_distance

        goal_state = GoalState(
            location=goal_loc,
            morphological_features=None,
            non_morphological_features=None,
            confidence=float(mlh.evidence),
            use_state=True,
            sender_id=getattr(lm, "learning_module_id", "LM_0"),
            sender_type="GSG",
            goal_tolerances=self.goal_tolerances,
        )

        self._current_goal = goal_state
        self._steps_since_goal = 0

        return [goal_state]

    def _random_exploration_goal(self, observations) -> list:
        """Generate a random exploration goal.

        Used when there are fewer than 2 hypotheses (e.g., during early
        inference or after all hypotheses are pruned).

        Args:
            observations: Current observations.

        Returns:
            List with one random GoalState, or empty list.
        """
        lm = self.parent_lm
        if lm is None:
            return []

        # Random direction from current location
        current_loc = None
        if hasattr(lm, "buffer") and lm.buffer.get_num_observations_on_object() > 0:
            try:
                current_loc = lm.buffer.get_current_location(
                    input_channel="first"
                )
            except (ValueError, IndexError):
                pass

        if current_loc is None:
            return []

        rng = np.random.default_rng()
        random_dir = rng.standard_normal(3)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
        goal_loc = current_loc + random_dir * self.desired_object_distance

        goal_state = GoalState(
            location=goal_loc,
            morphological_features=None,
            non_morphological_features=None,
            confidence=0.5,
            use_state=True,
            sender_id=getattr(lm, "learning_module_id", "LM_0"),
            sender_type="GSG",
            goal_tolerances=self.goal_tolerances,
        )

        return [goal_state]
