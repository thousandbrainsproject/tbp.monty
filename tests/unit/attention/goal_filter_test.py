# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.attention.goal_filter import (
    HardGoalFilter,
    NoopGoalFilter,
    SoftGoalFilter,
)
from tbp.monty.attention.voxel_grid import VOXEL_LEVELS, VoxelGrid

from .attention_system_test import goal_at

VOXEL_SIZE = 0.01

# Points inside the grid's attended, inhibited, and unoccupied voxels.
ATTENDED_POINT = [0.0, 0, 0]
INHIBITED_POINT = [VOXEL_SIZE, 0, 0]
OUTSIDE_POINT = [9.0, 9, 9]


def grid() -> VoxelGrid:
    """Build a grid with one attended and one inhibited voxel.

    Returns:
        A grid holding voxel (0,0,0) at weight 3 and (1,0,0) at weight -3.

    """
    frame = pd.DataFrame(
        {"weight": [3.0, -3.0]},
        index=pd.MultiIndex.from_tuples([(0, 0, 0), (1, 0, 0)], names=VOXEL_LEVELS),
    )
    return VoxelGrid(VOXEL_SIZE, frame)


class HardGoalFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = HardGoalFilter()
        self.grid = grid()

    def test_an_attended_goal_is_kept(self) -> None:
        goal = goal_at(ATTENDED_POINT)
        self.assertEqual(self.filter(self.grid, [goal]), [goal])

    def test_an_inhibited_goal_is_dropped(self) -> None:
        self.assertEqual(self.filter(self.grid, [goal_at(INHIBITED_POINT)]), [])

    def test_an_out_of_grid_goal_is_dropped(self) -> None:
        self.assertEqual(self.filter(self.grid, [goal_at(OUTSIDE_POINT)]), [])

    def test_an_unlocated_goal_passes_through(self) -> None:
        goal = goal_at(None)
        self.assertEqual(self.filter(self.grid, [goal]), [goal])

    def test_everything_passes_an_empty_grid(self) -> None:
        goals = [goal_at(ATTENDED_POINT), goal_at(OUTSIDE_POINT)]
        self.assertEqual(self.filter(VoxelGrid(VOXEL_SIZE), goals), goals)


class SoftGoalFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = SoftGoalFilter()
        self.grid = grid()

    def test_every_located_goal_is_kept(self) -> None:
        goals = [goal_at(ATTENDED_POINT), goal_at(OUTSIDE_POINT)]
        self.assertEqual(set(self.filter(self.grid, goals)), set(goals))

    def test_an_attended_goal_gains_confidence(self) -> None:
        goal = goal_at(ATTENDED_POINT)
        self.filter(self.grid, [goal])
        self.assertGreater(goal.confidence, 0.5)

    def test_an_inhibited_goal_loses_confidence(self) -> None:
        goal = goal_at(INHIBITED_POINT)
        self.filter(self.grid, [goal])
        self.assertLess(goal.confidence, 0.5)

    def test_an_out_of_grid_goal_is_neutral_by_default(self) -> None:
        # The default out_of_grid_weight of zero leaves confidence unchanged.
        goal = goal_at(OUTSIDE_POINT)
        self.filter(self.grid, [goal])
        self.assertEqual(goal.confidence, 0.5)

    def test_a_negative_out_of_grid_weight_punishes(self) -> None:
        goal = goal_at(OUTSIDE_POINT)
        SoftGoalFilter(out_of_grid_weight=-1.0)(self.grid, [goal])
        self.assertLess(goal.confidence, 0.5)

    def test_confidence_stays_within_bounds(self) -> None:
        certain = goal_at(ATTENDED_POINT)
        certain.confidence = 1.0
        hopeless = goal_at(INHIBITED_POINT)
        hopeless.confidence = 0.0
        self.filter(self.grid, [certain, hopeless])
        self.assertEqual(certain.confidence, 1.0)
        self.assertEqual(hopeless.confidence, 0.0)

    def test_goals_pass_untouched_through_an_empty_grid(self) -> None:
        goal = goal_at(ATTENDED_POINT)
        self.filter(VoxelGrid(VOXEL_SIZE), [goal])
        self.assertEqual(goal.confidence, 0.5)


class NoopGoalFilterTest(unittest.TestCase):
    def test_every_goal_passes_through_unchanged(self) -> None:
        goals = [goal_at(ATTENDED_POINT), goal_at(OUTSIDE_POINT), goal_at(None)]
        returned = NoopGoalFilter()(grid(), goals)
        self.assertEqual(returned, goals)
        self.assertEqual([g.confidence for g in returned], [0.5] * 3)


def one_voxel_grid(weight: float) -> VoxelGrid:
    """Build a grid whose only voxel is at the origin with the given weight.

    Returns:
        The one-voxel grid.

    """
    frame = pd.DataFrame(
        {"weight": [weight]},
        index=pd.MultiIndex.from_tuples([(0, 0, 0)], names=VOXEL_LEVELS),
    )
    return VoxelGrid(VOXEL_SIZE, frame)


class SoftGoalFilterPropertyTest(unittest.TestCase):
    """Reweighting must stay in bounds and move in the weight's direction.

    Property-based: hypothesis sweeps confidences and voxel weights,
    including the extremes and awkward in-betweens.
    """

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=-10.0, max_value=10.0),
    )
    def test_confidence_stays_in_bounds(self, confidence, weight) -> None:
        goal = goal_at([0.0, 0, 0])
        goal.confidence = confidence
        SoftGoalFilter()(one_voxel_grid(weight), [goal])
        self.assertGreaterEqual(goal.confidence, 0.0)
        self.assertLessEqual(goal.confidence, 1.0)

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=0.0, max_value=10.0),
    )
    def test_a_non_negative_weight_never_lowers_confidence(
        self, confidence, weight
    ) -> None:
        goal = goal_at([0.0, 0, 0])
        goal.confidence = confidence
        SoftGoalFilter()(one_voxel_grid(weight), [goal])
        self.assertGreaterEqual(goal.confidence, confidence)

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=-10.0, max_value=0.0),
    )
    def test_a_non_positive_weight_never_raises_confidence(
        self, confidence, weight
    ) -> None:
        goal = goal_at([0.0, 0, 0])
        goal.confidence = confidence
        SoftGoalFilter()(one_voxel_grid(weight), [goal])
        self.assertLessEqual(goal.confidence, confidence)

    @given(weight=st.floats(min_value=-10.0, max_value=10.0))
    def test_certainty_and_impossibility_are_fixed_points(self, weight) -> None:
        certain = goal_at([0.0, 0, 0])
        certain.confidence = 1.0
        hopeless = goal_at([0.0, 0, 0])
        hopeless.confidence = 0.0
        SoftGoalFilter()(one_voxel_grid(weight), [certain, hopeless])
        self.assertEqual(certain.confidence, 1.0)
        self.assertEqual(hopeless.confidence, 0.0)

    @given(
        low=st.floats(min_value=0.0, max_value=1.0),
        high=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=-10.0, max_value=10.0),
    )
    def test_reweighting_preserves_confidence_order(self, low, high, weight) -> None:
        # A more confident goal stays at least as confident as a less
        # confident one under the same voxel weight.
        low, high = sorted((low, high))
        low_goal = goal_at([0.0, 0, 0])
        low_goal.confidence = low
        high_goal = goal_at([0.0, 0, 0])
        high_goal.confidence = high
        SoftGoalFilter()(one_voxel_grid(weight), [low_goal, high_goal])
        self.assertLessEqual(low_goal.confidence, high_goal.confidence)


class HardGoalFilterOrderTest(unittest.TestCase):
    def test_kept_goals_preserve_their_relative_order(self) -> None:
        first = goal_at(ATTENDED_POINT)
        dropped = goal_at(OUTSIDE_POINT)
        second = goal_at(ATTENDED_POINT)
        returned = HardGoalFilter()(grid(), [first, dropped, second])
        self.assertEqual(returned, [first, second])

    def test_unlocated_goals_are_appended_after_kept_ones(self) -> None:
        unlocated = goal_at(None)
        kept = goal_at(ATTENDED_POINT)
        returned = HardGoalFilter()(grid(), [unlocated, kept])
        self.assertEqual(returned, [kept, unlocated])
