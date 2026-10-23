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
from hypothesis import given

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tests.strategies.arrays import float64_array
from tests.strategies.cmp import goals


class GraphLMProposeGoalsTest(unittest.TestCase):
    def test_returns_gsg_goals_when_last_obs_processed(self):
        pass

    def test_returns_no_goals_when_last_obs_not_processed(self):
        pass


class GraphLMProposeRegionTest(unittest.TestCase):
    @given(goals=goals(location_strategy=float64_array(shape=(3,)), min_goals=1))
    def test_returns_attention_region_that_contains_goal_locations_if_has_goals(
        self,
        goals: list[Goal],
    ) -> None:
        lm = GraphLM()
        lm._goals = goals

        proposed_region = lm.propose_region()

        for i, g in enumerate(goals):
            assert g.location is not None, "Goal location should not be None"
            np.testing.assert_array_equal(proposed_region.locations[i], g.location)
            self.assertEqual(proposed_region.weights[i], AttentionRegion.MAX_WEIGHT)

    def test_returns_empty_attention_region_if_has_no_goals(self) -> None:
        lm = GraphLM()
        lm._goals = []

        proposed_region = lm.propose_region()

        assert len(proposed_region) == 0
