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

from tbp.monty.attention.attention_system import (
    DEFAULT_VOXEL_SIZE,
    AttentionSystem,
    NoopAttentionSystem,
    max_pool,
    mean_pool,
    negative_priority_max_pool,
)
from tbp.monty.attention.decay import LinearDecay
from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, AttentionWeight, Goal

# The voxel coordinates are the source of truth; the test points below are
# constructed inside them, so the point/voxel correspondence holds by
# construction at any voxel size.
NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)


def point_in(voxel: tuple[int, int, int], offset: float = 0.5) -> np.ndarray:
    """Build a point inside the given voxel.

    Args:
        voxel: The (x, y, z) voxel coordinate.
        offset: Fractional position inside the voxel, in [0, 1).

    Returns:
        A point that voxelizes back to ``voxel``.
    """
    return (np.asarray(voxel) + offset) * DEFAULT_VOXEL_SIZE


# Two distinct points sharing the near voxel, and one in its own far voxel.
NEAR_POINTS = (point_in(NEAR_VOXEL, offset=0.25), point_in(NEAR_VOXEL, offset=0.75))
FAR_POINT = point_in(FAR_VOXEL)


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


def weight_at(location, weight: float = MAX_ATTENTION_WEIGHT) -> AttentionWeight:
    """Build an attention weight at the given location.

    Returns:
        An attention weight at the location carrying the given weight.

    """
    return AttentionWeight(
        location=np.asarray(location, dtype=float),
        weight=weight,
    )


def region(*locations, weight: float = MAX_ATTENTION_WEIGHT) -> list[AttentionWeight]:
    """Build a region from the given locations.

    Returns:
        One region: a list with one attention weight per location.

    """
    return [weight_at(location, weight=weight) for location in locations]


def weights_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], float]:
    """Map each occupied voxel to its weight.

    Returns:
        Voxel coordinate to weight, for every voxel the system holds.

    """
    data = system.voxel_grid.to_pandas()
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data["weight"].to_numpy().ravel().tolist()))


class AttentionSystemGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem()

    def test_locations_sharing_a_voxel_collapse_to_one_row(self) -> None:
        self.system.step([], [region(*NEAR_POINTS, FAR_POINT)])
        self.assertEqual(len(self.system.voxel_grid), 2)

    def test_no_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [])
        self.assertEqual(len(self.system.voxel_grid), 0)

    def test_empty_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [[], []])
        self.assertEqual(len(self.system.voxel_grid), 0)

    def test_a_step_adds_to_the_grid_rather_than_replacing_it(self) -> None:
        self.system.step([], [region(*NEAR_POINTS)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL])

    def test_regions_from_different_modules_merge_into_one_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS), region(FAR_POINT)])
        self.assertEqual(sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL])

    def test_reset_discards_the_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS)])
        self.system.reset()
        self.assertEqual(len(self.system.voxel_grid), 0)


class AttentionSystemWeightTest(unittest.TestCase):
    """Voxel weights decay toward zero and expire when they reach it."""

    def setUp(self) -> None:
        self.system = AttentionSystem(decay=LinearDecay(rate=0.1))

    def observe_near(self, weight: float = 3) -> None:
        """Propose the near voxel with the given weight."""
        self.system.step([], [region(NEAR_POINTS[0], weight=weight)])

    def observe_far(self) -> None:
        """Propose the far voxel."""
        self.system.step([], [region(FAR_POINT, weight=3)])

    def test_weight_starts_at_the_proposed_weight(self) -> None:
        self.observe_near()
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 3)

    def test_a_shared_voxel_takes_the_max_of_its_proposed_weights(self) -> None:
        proposals = [
            weight_at(NEAR_POINTS[0], weight=2),
            weight_at(NEAR_POINTS[1], weight=4),
        ]
        self.system.step([], [proposals])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 4)

    def test_a_negative_weight_dominates_a_shared_voxel(self) -> None:
        proposals = [
            weight_at(NEAR_POINTS[0], weight=4),
            weight_at(NEAR_POINTS[1], weight=-2),
        ]
        self.system.step([], [proposals])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -2)

    def test_an_unproposed_voxel_is_remembered(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(set(weights_by_voxel(self.system)), {NEAR_VOXEL, FAR_VOXEL})

    def test_an_unproposed_voxel_decays_by_the_decay_rate(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertAlmostEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.9)

    def test_a_negative_weight_decays_toward_zero(self) -> None:
        self.observe_near(weight=-3)
        self.observe_far()
        self.assertAlmostEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -2.9)

    def test_a_re_proposed_voxel_takes_the_fresh_weight(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0], weight=4)])
        self.system.step([], [region(NEAR_POINTS[0], weight=2)])
        # The fresh proposal replaces the remembered weight outright.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2)

    def test_a_voxel_expires_once_its_weight_decays_to_zero(self) -> None:
        # 0.15 steps to 0.05 on the first decay, within the rate of zero,
        # so it clamps and expires immediately.
        self.observe_near(weight=0.15)
        for _ in range(2):
            self.observe_far()
        self.assertEqual(set(weights_by_voxel(self.system)), {FAR_VOXEL})

    def test_proposing_nothing_still_decays_the_grid(self) -> None:
        self.observe_near()
        self.system.step([], [])
        self.assertAlmostEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.9)

    def test_the_grid_empties_once_everything_expires(self) -> None:
        self.observe_near(weight=0.15)
        for _ in range(2):
            self.system.step([], [])
        self.assertEqual(len(self.system.voxel_grid), 0)


class AttentionSystemFilterTest(unittest.TestCase):
    """Step returns only the goals that live in the updated voxel grid."""

    def setUp(self) -> None:
        self.system = AttentionSystem(decay=LinearDecay(rate=0.1))

    def test_only_goals_inside_the_grid_are_returned(self) -> None:
        inside = goal_at(NEAR_POINTS[0])
        outside = goal_at([9.0, 9, 9])
        returned = self.system.step([inside, outside], [region(*NEAR_POINTS, weight=3)])
        self.assertEqual(returned, [inside])

    def test_goals_are_filtered_against_the_updated_grid(self) -> None:
        # The region arrives on the same step as the goal it admits.
        goal = goal_at(NEAR_POINTS[1])
        self.assertEqual(
            self.system.step([goal], [region(NEAR_POINTS[0], weight=3)]), [goal]
        )

    def test_goals_pass_through_while_the_grid_is_empty(self) -> None:
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9])]
        self.assertEqual(self.system.step(goals, []), goals)

    def test_goals_without_a_location_pass_through(self) -> None:
        unlocated = goal_at(None)
        returned = self.system.step(
            [goal_at([9.0, 9, 9]), unlocated], [region(NEAR_POINTS[0], weight=3)]
        )
        self.assertEqual(returned, [unlocated])

    def test_a_remembered_voxel_still_admits_goals(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0], weight=3)])
        goal = goal_at(NEAR_POINTS[0])
        # The near voxel was not re-proposed, but has not expired either.
        self.assertEqual(
            self.system.step([goal], [region(FAR_POINT, weight=3)]), [goal]
        )

    def test_an_expired_voxel_no_longer_admits_goals(self) -> None:
        # 0.25 survives one decay (to 0.15); the second steps within the
        # rate of zero and clamps.
        self.system.step([], [region(NEAR_POINTS[0], weight=0.25)])
        goal = goal_at(NEAR_POINTS[0])
        for _ in range(2):
            self.system.step([], [region(FAR_POINT, weight=3)])
        # This step decays the near voxel to zero before filtering.
        self.assertEqual(self.system.step([goal], [region(FAR_POINT, weight=3)]), [])


class CustomPolicyInjectionTest(unittest.TestCase):
    """The decay and filter strategies are plain callables and swappable."""

    def test_a_custom_decay_is_used(self) -> None:
        system = AttentionSystem(decay=lambda _grid: None)
        system.step([], [region(NEAR_POINTS[0], weight=4)])
        system.step([], [])
        # The identity decay never erodes the weight.
        self.assertEqual(weights_by_voxel(system)[NEAR_VOXEL], 4)

    def test_a_custom_merge_is_used(self) -> None:
        # A merge that discards memory and keeps only the fresh proposal.
        system = AttentionSystem(merge=lambda _remembered, proposed: proposed)
        system.step([], [region(NEAR_POINTS[0], weight=4)])
        system.step([], [region(FAR_POINT, weight=4)])
        self.assertEqual(sorted(weights_by_voxel(system)), [FAR_VOXEL])

    def test_a_custom_goal_filter_is_used(self) -> None:
        system = AttentionSystem(goal_filter=lambda _grid, _goals: [])
        returned = system.step([goal_at(NEAR_POINTS[0])], [region(NEAR_POINTS[0])])
        self.assertEqual(returned, [])


class NoopAttentionSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = NoopAttentionSystem()

    def test_step_passes_every_goal_through(self) -> None:
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9]), goal_at(None)]
        self.assertEqual(self.system.step(goals, [region(NEAR_POINTS[0])]), goals)

    def test_reset_is_a_noop(self) -> None:
        self.system.reset()
        self.assertEqual(self.system.state_dict(), {})


class PoolFunctionTest(unittest.TestCase):
    """The built-in weight poolers."""

    def test_max_pool_takes_the_maximum(self) -> None:
        self.assertEqual(max_pool([2.0, -5.0, 4.0]), 4.0)

    def test_mean_pool_takes_the_mean(self) -> None:
        self.assertEqual(mean_pool([2.0, 4.0]), 3.0)

    def test_negative_priority_takes_the_max_when_all_non_negative(self) -> None:
        self.assertEqual(negative_priority_max_pool([0.0, 2.0, 4.0]), 4.0)

    def test_negative_priority_lets_any_negative_dominate(self) -> None:
        self.assertEqual(negative_priority_max_pool([4.0, -0.5, 2.0]), -0.5)

    def test_negative_priority_takes_the_most_negative(self) -> None:
        self.assertEqual(negative_priority_max_pool([-1.0, -3.0, -2.0]), -3.0)

    def test_a_single_weight_passes_through_every_pooler(self) -> None:
        for pool in (max_pool, mean_pool, negative_priority_max_pool):
            self.assertEqual(pool([7.0]), 7.0)


class AttentionSystemPipelineTest(unittest.TestCase):
    """Cross-cutting properties of a full step."""

    def setUp(self) -> None:
        self.system = AttentionSystem()

    def test_the_grid_index_is_sorted(self) -> None:
        # Proposals arrive in arbitrary order; the grid is kept sorted.
        self.system.step([], [region(FAR_POINT, NEAR_POINTS[0])])
        index = self.system.voxel_grid.index
        self.assertTrue(index.is_monotonic_increasing)

    def test_a_custom_pooler_is_used(self) -> None:
        system = AttentionSystem(pool_weights=sum)
        proposals = [
            weight_at(NEAR_POINTS[0], weight=2),
            weight_at(NEAR_POINTS[1], weight=3),
        ]
        system.step([], [proposals])
        self.assertEqual(weights_by_voxel(system)[NEAR_VOXEL], 5)

    def test_state_dict_carries_voxel_size_and_telemetry(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0])])
        state = self.system.state_dict()
        self.assertEqual(state["voxel_size"], DEFAULT_VOXEL_SIZE)
        self.assertEqual(len(state["voxel_grids"]), 1)

    def test_goals_and_regions_flow_through_one_step(self) -> None:
        # End to end: a region admits its goal, everything else is dropped.
        inside = goal_at(NEAR_POINTS[0])
        outside = goal_at([9.0, 9, 9])
        unlocated = goal_at(None)
        returned = self.system.step(
            [inside, outside, unlocated], [region(*NEAR_POINTS)]
        )
        self.assertEqual(returned, [inside, unlocated])
