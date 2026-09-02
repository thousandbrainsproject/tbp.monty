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

from tbp.monty.experiment.recognition_policy import (
    MaximumSteps,
    MaxTotalSteps,
    MinimumLMs,
    MontyIsDone,
    RecognitionCounter,
)
from tbp.monty.experiment.recognition_status import (
    RecognitionConclusion,
    RecognitionStatus,
)
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.monty_base import MontyBase


def _model_is_done(is_done: bool) -> MontyBase:
    model: MontyBase = MagicMock()
    model.is_done = is_done
    return model


def _model_with_conclusions(
    conclusions: list[RecognitionConclusion | None],
) -> MontyBase:
    learning_modules: list[LearningModule] = []
    for conclusion in conclusions:
        lm = MagicMock()
        lm.recognition_status = RecognitionStatus(conclusion=conclusion)
        learning_modules.append(lm)
    model: MontyBase = MagicMock()
    model.learning_modules = learning_modules
    return model


class MontyIsDoneTest(unittest.TestCase):
    @given(
        is_done=st.booleans(),
        max_steps=st.integers(min_value=0),
        step=st.integers(min_value=0),
    )
    def test_defers_to_model_even_with_max_steps(
        self, is_done: bool, max_steps: int, step: int
    ) -> None:
        # assume(step < max_steps)
        model = _model_is_done(is_done)
        policy = MontyIsDone()
        count = RecognitionCounter(step=step, max_steps=max_steps)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


class MaximumStepsTest(unittest.TestCase):
    @given(max_steps=st.integers(min_value=1), extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, max_steps: int, extra: int) -> None:
        model = _model_is_done(is_done=False)
        policy = MaximumSteps()
        count = RecognitionCounter(step=max_steps + extra, max_steps=max_steps)
        result = policy(model, count)
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
        model = _model_is_done(is_done)
        policy = MaximumSteps()
        count = RecognitionCounter(step=step, max_steps=max_steps)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


class MinimumCountTest(unittest.TestCase):
    @given(min_lms=st.integers(max_value=0))
    def test_raises_value_error_if_count_is_not_positive(self, min_lms: int) -> None:
        with self.assertRaises(ValueError):
            MinimumLMs(min_lms)

    @given(
        num_concluded=st.integers(min_value=0, max_value=10),
        num_pending=st.integers(min_value=0, max_value=10),
        min_lms=st.integers(min_value=1, max_value=10),
    )
    def test_done_iff_conclusion_count_reaches_count(
        self, num_concluded: int, num_pending: int, min_lms: int
    ) -> None:
        conclusions = [RecognitionConclusion.MATCH] * num_concluded + [
            None
        ] * num_pending
        model = _model_with_conclusions(conclusions)
        policy = MinimumLMs(min_lms)
        count = RecognitionCounter(step=0, max_steps=1)
        result = policy(model, count)
        self.assertEqual(result.is_done, num_concluded >= min_lms)

    def test_counts_any_conclusion_not_just_match(self) -> None:
        model = _model_with_conclusions(
            [RecognitionConclusion.NO_MATCH, RecognitionConclusion.TIME_OUT]
        )
        policy = MinimumLMs(2)
        count = RecognitionCounter(step=0, max_steps=1)
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, extra: int) -> None:
        model = _model_with_conclusions([None, None])
        policy = MinimumLMs(1)
        count = RecognitionCounter(step=10 + extra, max_steps=10)
        result = policy(model, count)
        self.assertTrue(result.is_done)


class MaxTotalStepsTest(unittest.TestCase):
    @given(max_total_steps=st.integers(min_value=1), extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_total_steps(
        self, max_total_steps: int, extra: int
    ) -> None:
        model = _model_is_done(is_done=False)
        policy = MaxTotalSteps(max_total_steps=max_total_steps)
        count = RecognitionCounter(step=max_total_steps + extra, max_steps=0)
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(
        is_done=st.booleans(),
        max_total_steps=st.integers(min_value=1),
        step=st.integers(min_value=0),
    )
    def test_defers_to_model_before_max_total_steps(
        self, is_done: bool, max_total_steps: int, step: int
    ) -> None:
        assume(step < max_total_steps)
        model = _model_is_done(is_done)
        policy = MaxTotalSteps(max_total_steps=max_total_steps)
        count = RecognitionCounter(step=step, max_steps=0)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)
