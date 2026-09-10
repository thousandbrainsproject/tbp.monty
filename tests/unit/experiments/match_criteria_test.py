# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase

from tbp.monty.experiment.match_criteria import AnyLMsMatch, NamedLMsMatch


class AnyLMsMatchTest(TestCase):
    def test_raises_value_error_if_count_is_not_positive(self) -> None:
        with self.assertRaises(ValueError):
            AnyLMsMatch(count=0)

        with self.assertRaises(ValueError):
            AnyLMsMatch(count=-1)

        AnyLMsMatch(count=1)

    def test_returns_true_if_match_terminal_states_count_equals_count(self) -> None:
        criterion = AnyLMsMatch(count=2)
        self.assertTrue(criterion({"lm1": "match", "lm2": "match"}))

    def test_returns_true_if_match_terminal_states_count_is_greater_than_count(
        self,
    ) -> None:
        criterion = AnyLMsMatch(count=2)
        self.assertTrue(criterion({"lm1": "match", "lm2": "match", "lm3": "match"}))

    def test_returns_false_if_match_terminal_states_count_is_less_than_count(
        self,
    ) -> None:
        criterion = AnyLMsMatch(count=2)
        self.assertFalse(criterion({"lm1": "match"}))


class NamedLMsMatchTest(TestCase):
    def test_raises_value_error_if_ids_is_empty(self) -> None:
        with self.assertRaises(ValueError):
            NamedLMsMatch(ids=[])

        NamedLMsMatch(ids=["lm1"])

    def test_returns_true_if_all_ids_are_match(self) -> None:
        criterion = NamedLMsMatch(ids=["lm1", "lm2"])
        self.assertTrue(criterion({"lm1": "match", "lm2": "match"}))

    def test_returns_false_if_not_all_ids_are_match(self) -> None:
        criterion = NamedLMsMatch(ids=["lm1", "lm2"])
        self.assertFalse(criterion({"lm1": "match", "lm2": "no_match"}))
