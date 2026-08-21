# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from tbp.monty.attention.voxel_grid import VoxelGrid
from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, Goal


class GoalFilter(Protocol):
    def __call__(self, voxel_grid: VoxelGrid, goals: Sequence[Goal]) -> list[Goal]: ...


class NoopGoalFilter(GoalFilter):
    """Pass every goal through unchanged."""

    def __call__(
        self,
        voxel_grid: VoxelGrid,  # noqa: ARG002
        goals: Sequence[Goal],
    ) -> list[Goal]:
        """Return the goals unchanged.

        Args:
            voxel_grid: Unused.
            goals: The goals to (not) filter.

        Returns:
            The goals, unfiltered.
        """
        return list(goals)


class HardGoalFilter(GoalFilter):
    """Keep only the goals inside a non-negatively weighted voxel.

    Goals outside the grid or in an inhibited (negative-weight) voxel are
    dropped. Goals without a location pass through, as does everything when
    the grid is empty.
    """

    def __call__(self, voxel_grid: VoxelGrid, goals: Sequence[Goal]) -> list[Goal]:
        """Filter the goals against the grid.

        Args:
            voxel_grid: The current voxel grid.
            goals: The goals to filter.

        Returns:
            The goals inside a non-negatively weighted voxel, plus any
            without a location. All goals, if the grid is empty.
        """
        if len(voxel_grid) == 0 or len(goals) == 0:
            return list(goals)

        located = [g for g in goals if g.location is not None]
        unlocated = [g for g in goals if g.location is None]
        if not located:
            return unlocated

        points = np.array([g.location for g in located])
        voxel_weights = voxel_grid.feature_at_points("weight", points)

        # Out-of-grid lookups are NaN, and NaN comparisons are False, so
        # out-of-grid goals are dropped along with the inhibited ones.
        kept = [
            goal
            for goal, voxel_weight in zip(located, voxel_weights)
            if voxel_weight >= 0
        ]
        return kept + unlocated


class SoftGoalFilter(GoalFilter):
    """Keep every goal, reweighting confidence by its voxel's weight.

    Attended (positive-weight) voxels boost a goal's confidence toward one;
    inhibited voxels and out-of-grid locations push it toward zero. The
    confidence update is done in place on the goal.
    """

    def __init__(
        self,
        out_of_grid_weight: float = 0.0,
        scale: float = MAX_ATTENTION_WEIGHT / 3,
        gain: float = 2.0,
    ) -> None:
        """Initialize the filter.

        Args:
            out_of_grid_weight: The weight treated as standing in for a goal
                outside the grid.
            scale: How much voxel weight it takes to saturate the confidence
                boost or suppression (see ``_signed_sigmoid``).
            gain: Distortion ceiling: the strongest boost is the gain-th
                root of the confidence, the strongest suppression its
                gain-th power.
        """
        self._out_of_grid_weight = out_of_grid_weight
        self._scale = scale
        self._gain = gain

    def __call__(self, voxel_grid: VoxelGrid, goals: Sequence[Goal]) -> list[Goal]:
        """Reweight the goals against the grid.

        Args:
            voxel_grid: The current voxel grid.
            goals: The goals to reweight.

        Returns:
            Every goal, with located goals' confidences reweighted. All
            goals untouched, if the grid is empty.
        """
        if len(voxel_grid) == 0 or len(goals) == 0:
            return list(goals)

        located = [g for g in goals if g.location is not None]
        unlocated = [g for g in goals if g.location is None]
        if not located:
            return unlocated

        points = np.array([g.location for g in located])
        voxel_weights = voxel_grid.feature_at_points("weight", points)
        voxel_weights = np.where(
            np.isnan(voxel_weights), self._out_of_grid_weight, voxel_weights
        )
        confidences = np.array([g.confidence for g in located])
        weighted_confidences = self._weighted(confidences, voxel_weights)

        for goal, new_confidence in zip(located, weighted_confidences):
            goal.confidence = float(new_confidence)

        return located + unlocated

    def _weighted(
        self,
        confidence: npt.NDArray[np.floating],
        voxel_weight: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Reweight confidences by their voxel weights.

        The tanh squashes voxel weights into (-1, 1), roughly linear near
        zero and saturating once their magnitude is well past the scale.
        Confidence stays in [0, 1]: the exponent ``gain ** -tanh`` lies in
        (1/gain, gain), boosting toward one for positive voxel weights and
        suppressing toward zero for negative ones. Zero and one are fixed
        points.

        Args:
            confidence: The confidences to reweight, in [0, 1].
            voxel_weight: The voxel weight at each goal's location.

        Returns:
            The reweighted confidences.
        """
        squashed = np.tanh(voxel_weight / (2.0 * self._scale))
        return confidence ** (self._gain**-squashed)
