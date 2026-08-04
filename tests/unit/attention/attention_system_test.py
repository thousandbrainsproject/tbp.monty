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

import numpy as np

from tbp.monty.attention.attention_system import AttentionSystem
from tbp.monty.cmp import Goal

# Two points inside one voxel, and a third far enough away to occupy its own.
NEAR_POINTS = ([0.0, 0, 0], [0.005, 0, 0])
FAR_POINT = [0.5, 0, 0]
# The voxels those points fall in, at voxel_size=0.01.
NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)


def goal_at(location) -> Goal:
    """Build a goal at the given location.

    Returns:
        A goal whose only meaningful property here is its location.

    """
    return Goal(
        location=None if location is None else np.asarray(location, dtype=float),
        morphological_features=None,
        non_morphological_features=None,
        confidence=0.5,
        use_state=False,
        sender_id="SM_0",
        sender_type="SM",
        goal_tolerances=None,
    )


def region(*locations) -> list[Goal]:
    """Build a region from the given locations.

    Returns:
        One region: a list with one goal per location.

    """
    return [goal_at(location) for location in locations]


def column_by_voxel(
    system: AttentionSystem, column: str
) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to one of its column values.

    Returns:
        Voxel coordinate to value, for every voxel the system holds.

    """
    data = system.grid
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data[column].to_numpy().ravel().tolist()))


def ages_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to its remaining age.

    Returns:
        Voxel coordinate to age, for every voxel the system holds.

    """
    return column_by_voxel(system, "age")


def counts_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to how many times it has been observed.

    Returns:
        Voxel coordinate to count, for every voxel the system holds.

    """
    return column_by_voxel(system, "count")


class AttentionSystemGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)

    def test_locations_sharing_a_voxel_collapse_to_one_row(self) -> None:
        self.system.step([], [region(*NEAR_POINTS, FAR_POINT)])
        self.assertEqual(len(self.system.grid), 2)

    def test_no_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)

    def test_empty_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [[], []])
        self.assertEqual(len(self.system.grid), 0)

    def test_goals_without_a_location_are_not_voxelized(self) -> None:
        self.system.step([], [[goal_at(None)]])
        self.assertEqual(len(self.system.grid), 0)

    def test_a_step_adds_to_the_grid_rather_than_replacing_it(self) -> None:
        self.system.step([], [region(*NEAR_POINTS)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(
            sorted(ages_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL]
        )

    def test_regions_from_different_modules_merge_into_one_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS), region(FAR_POINT)])
        self.assertEqual(
            sorted(ages_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL]
        )

    def test_reset_discards_the_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS)])
        self.system.reset()
        self.assertEqual(len(self.system.grid), 0)


class AttentionSystemAgeTest(unittest.TestCase):
    """Voxels persist across steps, ageing until they are re-observed or expire."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, lifetime=3)

    def observe_near(self) -> None:
        """Observe the near voxel."""
        self.system.step([], [region(NEAR_POINTS[0])])

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.system.step([], [region(FAR_POINT)])

    def test_age_starts_at_the_full_lifetime(self) -> None:
        self.observe_near()
        self.assertEqual(ages_by_voxel(self.system)[NEAR_VOXEL], 3)

    def test_age_is_stored_as_an_integer(self) -> None:
        self.observe_near()
        self.assertEqual(self.system.grid["age"].to_numpy().dtype, np.int32)

    def test_lifetime_is_exposed(self) -> None:
        self.assertEqual(AttentionSystem(lifetime=9).lifetime, 9)

    def test_lifetime_must_be_positive(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                AttentionSystem(lifetime=bad)

    def test_an_unobserved_voxel_is_remembered(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(set(ages_by_voxel(self.system)), {NEAR_VOXEL, FAR_VOXEL})

    def test_an_unobserved_voxel_ages_by_one_step(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(ages_by_voxel(self.system)[NEAR_VOXEL], 2)

    def test_a_re_observed_voxel_returns_to_a_full_age(self) -> None:
        self.observe_near()
        self.observe_far()
        self.observe_near()
        self.assertEqual(ages_by_voxel(self.system)[NEAR_VOXEL], 3)

    def test_a_voxel_expires_after_its_lifetime_of_steps(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.observe_far()
        self.assertEqual(set(ages_by_voxel(self.system)), {FAR_VOXEL})

    def test_observing_nothing_still_ages_the_grid(self) -> None:
        self.observe_near()
        self.system.step([], [])
        self.assertEqual(ages_by_voxel(self.system)[NEAR_VOXEL], 2)

    def test_the_grid_empties_once_everything_expires(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)

    def test_age_stays_an_integer_across_steps(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(self.system.grid["age"].to_numpy().dtype, np.int32)


class AttentionSystemCountTest(unittest.TestCase):
    """Count tallies how many times a voxel has been observed."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, lifetime=3)

    def observe_near(self) -> None:
        """Observe the near voxel."""
        self.system.step([], [region(NEAR_POINTS[0])])

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.system.step([], [region(FAR_POINT)])

    def test_a_newly_seen_voxel_starts_at_one(self) -> None:
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 1)

    def test_re_observing_adds_to_the_count(self) -> None:
        for expected in (1, 2, 3):
            self.observe_near()
            self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], expected)

    def test_an_unobserved_voxel_keeps_its_count(self) -> None:
        self.observe_near()
        self.observe_near()
        self.observe_far()
        counts = counts_by_voxel(self.system)
        self.assertEqual(counts[NEAR_VOXEL], 2)
        self.assertEqual(counts[FAR_VOXEL], 1)

    def test_counts_are_tallied_independently_per_voxel(self) -> None:
        self.observe_near()
        self.observe_far()
        self.observe_far()
        counts = counts_by_voxel(self.system)
        self.assertEqual(counts[NEAR_VOXEL], 1)
        self.assertEqual(counts[FAR_VOXEL], 2)

    def test_each_region_seeing_a_voxel_contributes_a_sighting(self) -> None:
        # Two modules propose overlapping regions on the same step.
        self.system.step([], [region(NEAR_POINTS[0]), region(NEAR_POINTS[1])])
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 2)

    def test_count_restarts_after_a_voxel_expires(self) -> None:
        # An expired voxel is forgotten, so its tally starts over.
        self.observe_near()
        self.observe_near()
        for _ in range(3):
            self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 1)

    def test_count_stays_an_integer_across_steps(self) -> None:
        self.observe_near()
        self.observe_near()
        self.assertEqual(self.system.grid["count"].to_numpy().dtype, np.int32)


class AttentionSystemContainsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)
        self.system.step([], [region(*NEAR_POINTS, FAR_POINT)])

    def test_many_locations_yield_an_array(self) -> None:
        result = self.system.contains_points(np.array([[0.0, 0, 0], [9.0, 9, 9]]))
        np.testing.assert_array_equal(result, [True, False])

    def test_a_single_flat_point_is_accepted(self) -> None:
        # Normalized to (1, 3), so the result is still an array.
        np.testing.assert_array_equal(
            self.system.contains_points(np.array([0.0, 0, 0])), [True]
        )

    def test_any_location_in_an_occupied_voxel_is_contained(self) -> None:
        # A different point in the same voxel as an observed one.
        np.testing.assert_array_equal(
            self.system.contains_points(np.array([[0.009, 0, 0]])), [True]
        )

    def test_an_empty_grid_contains_nothing(self) -> None:
        empty = AttentionSystem(voxel_size=0.01)
        np.testing.assert_array_equal(
            empty.contains_points(np.array([[0.0, 0, 0]])), [False]
        )


class AttentionSystemFilterTest(unittest.TestCase):
    """Step returns only the goals that live in the updated voxel grid."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, lifetime=3)

    def test_only_goals_inside_the_grid_are_returned(self) -> None:
        inside = goal_at(NEAR_POINTS[0])
        outside = goal_at([9.0, 9, 9])
        returned = self.system.step(
            [inside, outside], [region(*NEAR_POINTS)]
        )
        self.assertEqual(returned, [inside])

    def test_goals_are_filtered_against_the_updated_grid(self) -> None:
        # The region arrives on the same step as the goal it admits.
        goal = goal_at(NEAR_POINTS[1])
        self.assertEqual(
            self.system.step([goal], [region(NEAR_POINTS[0])]), [goal]
        )

    def test_goals_pass_through_while_the_grid_is_empty(self) -> None:
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9])]
        self.assertEqual(self.system.step(goals, []), goals)

    def test_goals_without_a_location_pass_through(self) -> None:
        unlocated = goal_at(None)
        returned = self.system.step(
            [goal_at([9.0, 9, 9]), unlocated], [region(NEAR_POINTS[0])]
        )
        self.assertEqual(returned, [unlocated])

    def test_a_remembered_voxel_still_admits_goals(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0])])
        goal = goal_at(NEAR_POINTS[0])
        # The near voxel was not re-observed, but has not expired either.
        self.assertEqual(self.system.step([goal], [region(FAR_POINT)]), [goal])

    def test_an_expired_voxel_no_longer_admits_goals(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0])])
        goal = goal_at(NEAR_POINTS[0])
        for _ in range(2):
            self.system.step([], [region(FAR_POINT)])
        # This step ages the near voxel past its lifetime before filtering.
        self.assertEqual(self.system.step([goal], [region(FAR_POINT)]), [])
