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
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest

from tbp.monty.cmp import MAX_ATTENTION_WEIGHT
from tbp.monty.frameworks.models.evidence_matching.region_proposal.excite_goal_locations import (  # noqa: E501
    ExciteGoalLocations,
    sample_ball,
)
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionContext,
)


def goal_at(location: np.ndarray | None) -> MagicMock:
    goal = MagicMock()
    goal.location = location
    return goal


def context_with_goals(*goals: MagicMock) -> MagicMock:
    # The proposer reads nothing else off the context.
    context = MagicMock(spec=RegionContext)
    context.goals = goals
    return context


class SampleBallTest(unittest.TestCase):
    def test_contains_the_center(self) -> None:
        center = np.array([1.0, -2.0, 3.0])

        points = sample_ball(center, radius=0.01, lattice_steps=4)

        self.assertTrue((points == center).all(axis=1).any())

    def test_every_point_lies_within_the_radius(self) -> None:
        center = np.array([1.0, -2.0, 3.0])

        points = sample_ball(center, radius=0.01, lattice_steps=4)

        nptest.assert_array_less(np.linalg.norm(points - center, axis=1), 0.01 + 1e-12)

    def test_spans_the_full_diameter(self) -> None:
        points = sample_ball(np.zeros(3), radius=0.01, lattice_steps=4)

        self.assertAlmostEqual(points[:, 0].min(), -0.01)
        self.assertAlmostEqual(points[:, 0].max(), 0.01)


class ExciteGoalLocationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposer = ExciteGoalLocations()

    def test_proposes_nothing_without_goals(self) -> None:
        self.assertIsNone(self.proposer(context_with_goals()))

    def test_proposes_nothing_when_no_goal_has_a_location(self) -> None:
        context = context_with_goals(goal_at(None), goal_at(None))

        self.assertIsNone(self.proposer(context))

    def test_excites_a_ball_around_a_goal_location(self) -> None:
        location = np.array([0.1, 0.2, 0.3])

        region = self.proposer(context_with_goals(goal_at(location)))

        distances = np.linalg.norm(region.locations - location, axis=1)
        nptest.assert_array_less(distances, 0.01 + 1e-12)
        nptest.assert_array_equal(
            region.weights, np.full(len(region), MAX_ATTENTION_WEIGHT)
        )

    def test_skips_goals_without_a_location(self) -> None:
        location = np.array([0.1, 0.2, 0.3])
        context = context_with_goals(goal_at(None), goal_at(location))

        region = self.proposer(context)

        distances = np.linalg.norm(region.locations - location, axis=1)
        nptest.assert_array_less(distances, 0.01 + 1e-12)

    def test_proposes_one_ball_per_located_goal(self) -> None:
        near = np.zeros(3)
        far = np.array([1.0, 0.0, 0.0])

        region = self.proposer(context_with_goals(goal_at(near), goal_at(far)))

        to_near = np.linalg.norm(region.locations - near, axis=1) <= 0.01 + 1e-12
        to_far = np.linalg.norm(region.locations - far, axis=1) <= 0.01 + 1e-12
        self.assertTrue(to_near.any())
        self.assertTrue(to_far.any())
        self.assertTrue((to_near | to_far).all())

    def test_radius_and_weight_are_parameterized(self) -> None:
        proposer = ExciteGoalLocations(radius=0.05, weight=0.5)

        region = proposer(context_with_goals(goal_at(np.zeros(3))))

        self.assertAlmostEqual(np.linalg.norm(region.locations, axis=1).max(), 0.05)
        nptest.assert_array_equal(region.weights, np.full(len(region), 0.5))
