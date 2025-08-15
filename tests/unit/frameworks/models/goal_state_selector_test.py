# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import unittest
from re import A

import numpy as np

from tbp.monty.frameworks.models.goal_state_selector import (
    GoalStateSelector,
    flatten,
    sort_states_by_confidence,
)
from tbp.monty.frameworks.models.states import GoalState, State


class FlattenTest(unittest.TestCase):
    def test_flat(self) -> None:
        a = make_dummy_state(use_state=False)
        b = make_dummy_goal_state(use_state=False)
        self.assertEqual(flatten([]), [])
        self.assertEqual(flatten([a]), [a])
        self.assertEqual(flatten([a, b]), [a, b])

    def test_nested(self) -> None:
        flat = [
            None,
            make_dummy_state(use_state=False),
            make_dummy_goal_state(use_state=False),
            None,
            make_dummy_state(use_state=False),
            make_dummy_goal_state(use_state=False),
        ]
        nested = [flat[0], [flat[1], flat[2]], [[flat[3], flat[4], flat[5]]]]
        self.assertEqual(flatten(nested), flat)


class SortStatesByConfidenceTest(unittest.TestCase):

    def test_empty(self) -> None:
        """Test that empty iterables are sorted to an empty list."""
        self.assertEqual(sort_states_by_confidence([]), [])

    def test_basic(self) -> None:
        """Test that states are sorted by confidence."""
        a = make_dummy_state(confidence=0.0)
        b = make_dummy_state(confidence=1.0)
        self.assertEqual(sort_states_by_confidence([a, b]), [a, b])

    def test_reverse(self) -> None:
        """Test that reverse=True puts higher-confidence states first."""
        a = make_dummy_state(confidence=0.0)
        b = make_dummy_state(confidence=1.0)
        self.assertEqual(sort_states_by_confidence([a, b], reverse=True), [b, a])

    def test_none_and_nan(self) -> None:
        """Test that None and np.nan values are sorted w/ lowest possible confidence."""
        a = make_dummy_state(confidence=None, use_state=False)
        b = make_dummy_state(confidence=0.0)
        self.assertEqual(sort_states_by_confidence([a, b]), [a, b])
        self.assertEqual(sort_states_by_confidence([a, b], reverse=True), [b, a])

        a = make_dummy_state(confidence=np.nan, use_state=False)
        b = make_dummy_state(confidence=0.0)
        self.assertEqual(sort_states_by_confidence([a, b]), [a, b])
        self.assertEqual(sort_states_by_confidence([a, b], reverse=True), [b, a])


class GoalStateSelectorTest(unittest.TestCase):

    def test_select_empty_list(self) -> None:
        """Test that empty iterables are sorted to an empty list."""
        gss = GoalStateSelector()
        self.assertEqual(gss.select([]), None)

    def test_select_no_valid_goal_states_returns_none(self) -> None:
        """Test that empty iterables are sorted to an empty list."""
        goal_states = [
            None,
            make_dummy_goal_state(use_state=False),
        ]
        gss = GoalStateSelector()
        self.assertEqual(gss.select(goal_states), None)

    def test_select_returns_goal_state_with_highest_confidence(self) -> None:
        """Test that the goal state with the highest confidence is returned."""
        a = make_dummy_goal_state(confidence=0.0)
        b = make_dummy_goal_state(confidence=1.0)
        gss = GoalStateSelector()
        self.assertEqual(gss.select([a, b]), b)

    def test_select_nested_heterogeneous_input(self) -> None:
        """Test that the goal state with the highest confidence is returned.

        This test is more realistic than the basic test, and tests that the
        selector can handle nested, heterogeneous input.
        """
        gs_low = make_dummy_goal_state(confidence=0.0)
        gs_medium = make_dummy_goal_state(confidence=0.5)
        gs_high = make_dummy_goal_state(confidence=1.0)
        gs_none = make_dummy_goal_state(confidence=None, use_state=False)
        gs_nan = make_dummy_goal_state(confidence=np.nan, use_state=False)

        goal_states = [
            None,
            gs_low,
            [
                gs_none,
                gs_medium,
                [
                    gs_nan,
                    gs_high,
                ]
            ]
        ]
        gss = GoalStateSelector()
        self.assertEqual(gss.select(goal_states), gs_high)


def make_dummy_state(**kwargs) -> State:
    defaults = {
        "location": np.zeros(3),
        "morphological_features": {
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        "non_morphological_features": None,
        "confidence": 1.0,
        "use_state": True,
        "sender_id": "sender_id",
        "sender_type": "SM",
    }
    init_args = {**defaults, **kwargs}
    return State(**init_args)


def make_dummy_goal_state(**kwargs) -> GoalState:
    defaults = {
        "location": np.zeros(3),
        "morphological_features": None,
        "non_morphological_features": None,
        "confidence": 1.0,
        "use_state": True,
        "sender_id": "sender_id",
        "sender_type": "GSG",
        "goal_tolerances": None,
        "info": None,
    }
    init_args = {**defaults, **kwargs}
    return GoalState(**init_args)


if __name__ == "__main__":
    unittest.main()

