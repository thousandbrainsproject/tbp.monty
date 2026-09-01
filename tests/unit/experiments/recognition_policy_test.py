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
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.monty_base import MontyBase


def _an_experiment(max_steps: int = 1) -> MontyExperiment:
    exp: MontyExperiment = MagicMock()
    exp.max_steps = max_steps
    return exp


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
    @given(max_steps=st.integers(min_value=1), extra_steps=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(
        self, max_steps: int, extra_steps: int
    ) -> None:
        exp = _an_experiment(max_steps)
        model=_model_is_done(False)
        policy = MontyIsDone()
        result = policy(exp, model, max_steps + extra_steps)
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
        exp = _an_experiment(max_steps)
        model=_model_is_done(is_done)
        policy = MontyIsDone()
        result = policy(exp, model, step)
        self.assertEqual(result.is_done, is_done)


class MinimumCountTest(unittest.TestCase):
    @given(count=st.integers(max_value=0))
    def test_raises_value_error_if_count_is_not_positive(self, count: int) -> None:
        with self.assertRaises(ValueError):
            MinimumCount(count=count)

    @given(
        num_concluded=st.integers(min_value=0, max_value=10),
        num_pending=st.integers(min_value=0, max_value=10),
        count=st.integers(min_value=1, max_value=10),
    )
    def test_done_iff_conclusion_count_reaches_count(
        self, num_concluded: int, num_pending: int, count: int
    ) -> None:
        exp = _an_experiment()
        conclusions = [RecognitionConclusion.MATCH] * num_concluded + [
            None
        ] * num_pending
        model = _model_with_conclusions(conclusions)
        policy = MinimumCount(count=count)
        result = policy(exp, model, 0)
        self.assertEqual(result.is_done, num_concluded >= count)

    def test_counts_any_conclusion_not_just_match(self) -> None:
        exp = _an_experiment()
        model = _model_with_conclusions(
            [RecognitionConclusion.NO_MATCH, RecognitionConclusion.TIME_OUT]
        )
        policy = MinimumCount(count=2)
        result = policy(exp, model, 0)
        self.assertTrue(result.is_done)

    @given(extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, extra: int) -> None:
        exp = _an_experiment(10)
        model = _model_with_conclusions([None, None])
        policy = MinimumCount(count=1)
        result = policy(exp, model, 10 + extra)
        self.assertTrue(result.is_done)
