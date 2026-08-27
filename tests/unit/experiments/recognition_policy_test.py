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

from hypothesis import assume, given
from hypothesis import strategies as st

from tbp.monty.experiment.recognition_policy import MinimumCount, MontyIsDone
from tbp.monty.experiment.recognition_status import (
    RecognitionConclusion,
    RecognitionStatus,
)


def _model_is_done(is_done: bool) -> MagicMock:
    model = MagicMock()
    model.is_done = is_done
    return model


def _model_with_conclusions(
    conclusions: list[RecognitionConclusion | None],
) -> MagicMock:
    learning_modules = []
    for conclusion in conclusions:
        lm = MagicMock()
        lm.recognition_status = RecognitionStatus(conclusion=conclusion)
        learning_modules.append(lm)
    model = MagicMock()
    model.learning_modules = learning_modules
    return model


class MontyIsDoneTest(unittest.TestCase):
    def test_raises_if_max_steps_is_not_positive(self) -> None:
        with self.assertRaises(ValueError):
            MontyIsDone(max_steps=0)

        with self.assertRaises(ValueError):
            MontyIsDone(max_steps=-1)

    @given(is_done=st.booleans(), step=st.integers(min_value=0))
    def test_mirrors_model_when_no_max_steps(self, is_done: bool, step: int) -> None:
        policy = MontyIsDone(max_steps=None)
        result = policy(model=_model_is_done(is_done), step=step)
        self.assertEqual(result.is_done, is_done)

    @given(max_steps=st.integers(min_value=1), extra_steps=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(
        self, max_steps: int, extra_steps: int
    ) -> None:
        policy = MontyIsDone(max_steps=max_steps)
        result = policy(
            model=_model_is_done(is_done=False), step=max_steps + extra_steps
        )
        self.assertTrue(result.is_done)

    @given(
        is_done=st.booleans(),
        max_steps=st.integers(min_value=1),
        step=st.integers(min_value=0),
    )
    def test_defers_to_model_before_max_steps(
        self, is_done: bool, max_steps: int, step: int
    ) -> None:
        assume(step < max_steps)
        policy = MontyIsDone(max_steps=max_steps)
        result = policy(model=_model_is_done(is_done), step=step)
        self.assertEqual(result.is_done, is_done)


class MinimumCountTest(unittest.TestCase):
    def test_raises_value_error_if_count_is_not_positive(self) -> None:
        with self.assertRaises(ValueError):
            MinimumCount(count=0, max_steps=10)

        with self.assertRaises(ValueError):
            MinimumCount(count=-1, max_steps=10)

    def test_raises_value_error_if_max_steps_is_not_positive(self) -> None:
        with self.assertRaises(ValueError):
            MinimumCount(count=1, max_steps=0)

        with self.assertRaises(ValueError):
            MinimumCount(count=1, max_steps=-1)

    @given(
        num_concluded=st.integers(min_value=0, max_value=10),
        num_pending=st.integers(min_value=0, max_value=10),
        count=st.integers(min_value=1, max_value=10),
    )
    def test_done_iff_conclusion_count_reaches_count(
        self, num_concluded: int, num_pending: int, count: int
    ) -> None:
        policy = MinimumCount(count=count, max_steps=10)
        conclusions = [RecognitionConclusion.MATCH] * num_concluded + [
            None
        ] * num_pending
        model = _model_with_conclusions(conclusions)
        self.assertEqual(policy(model=model, step=0).is_done, num_concluded >= count)

    def test_counts_any_conclusion_not_just_match(self) -> None:
        policy = MinimumCount(count=2, max_steps=10)
        model = _model_with_conclusions(
            [RecognitionConclusion.NO_MATCH, RecognitionConclusion.TIME_OUT]
        )
        self.assertTrue(policy(model=model, step=0).is_done)

    @given(extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, extra: int) -> None:
        policy = MinimumCount(count=1, max_steps=10)
        model = _model_with_conclusions([None, None])
        self.assertTrue(policy(model=model, step=10 + extra).is_done)
