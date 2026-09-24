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

from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.experiment.recognition_policy import (
    AnyPolicy,
    MaximumSteps,
    MaxTotalSteps,
    MinimumLMs,
    MontyIsDone,
    NaiveScan,
    ObjectRecognition,
    RecognitionCounter,
    RecognitionPolicy,
    RecognitionResult,
)
from tbp.monty.experiment.recognition_status import (
    RecognitionConclusion,
    RecognitionStatus,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule


@st.composite
def ascending_ints(draw: st.DrawFn, min_value: int):
    a = draw(st.integers(min_value=min_value))
    b = draw(st.integers(min_value=a + 1))
    return (a, b)


def _model_is_done(is_done: bool) -> MagicMock:
    model = MagicMock()
    model.is_done = is_done
    return model


def _model_with_conclusions(
    conclusions: list[RecognitionConclusion | None],
) -> MagicMock:
    learning_modules: list[LearningModule] = []
    for conclusion in conclusions:
        lm = MagicMock()
        lm.recognition_status = RecognitionStatus(conclusion=conclusion)
        learning_modules.append(lm)
    model = MagicMock()
    model.learning_modules = learning_modules
    return model


def _model_with_recognition(
    is_done: bool, is_exploring: bool, matching_steps: int
) -> MagicMock:
    model = MagicMock()
    model.is_done = is_done
    model.is_exploring = is_exploring
    model.matching_steps = matching_steps
    return model


class MontyIsDoneTest(unittest.TestCase):
    @given(
        is_done=st.booleans(),
        step=st.integers(min_value=0),
    )
    def test_defers_to_model_regardless_of_step(self, is_done: bool, step: int) -> None:
        model = _model_is_done(is_done)
        policy = MontyIsDone()
        count = RecognitionCounter(step)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


class MaxTotalStepsTest(unittest.TestCase):
    @given(max_total_steps=st.integers(max_value=0))
    def test_raises_value_error_if_max_total_steps_is_not_positive(
        self, max_total_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MaxTotalSteps(max_total_steps)

    @given(
        step=st.integers(min_value=0),
        max_total_steps=st.integers(min_value=1),
    )
    def test_times_out_at_or_after_max_total_steps(
        self, step: int, max_total_steps: int
    ) -> None:
        model = MagicMock()
        policy = MaxTotalSteps(max_total_steps)
        count = RecognitionCounter(step)
        result = policy(model, count)
        is_done = step >= max_total_steps
        self.assertEqual(result.is_done, is_done)
        model.assert_not_called()


class MaximumStepsTest(unittest.TestCase):
    @given(
        max_train_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_train_steps_is_not_positive(
        self, max_train_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MaximumSteps(max_train_steps, 1)

    @given(
        max_eval_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_eval_steps_is_not_positive(
        self, max_eval_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MaximumSteps(1, max_eval_steps)

    @given(
        is_done=st.booleans(),
        matching_steps=st.integers(min_value=0),
        max_steps=st.integers(min_value=1),
    )
    def test_times_out_at_or_after_max_steps(
        self, is_done: bool, matching_steps: int, max_steps: int
    ) -> None:
        model = _model_with_recognition(
            is_done, is_exploring=False, matching_steps=matching_steps
        )
        policy = MaximumSteps(max_steps, max_steps)
        count = RecognitionCounter()
        result = policy(model, count)
        is_done = matching_steps >= max_steps
        self.assertEqual(result.is_done, is_done)

    @given(
        is_done=st.booleans(),
        matching_steps=st.integers(min_value=0),
        max_steps=st.integers(min_value=1),
    )
    def test_no_timeout_if_is_exploring(
        self,
        is_done: bool,
        matching_steps: int,
        max_steps: int,
    ) -> None:
        model = _model_with_recognition(
            is_done=is_done, is_exploring=True, matching_steps=matching_steps
        )
        policy = MaximumSteps(max_steps, max_steps)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertFalse(result.is_done)

    @given(
        matching_steps=st.integers(min_value=0),
        mode=st.sampled_from(ExperimentMode),
        max_train_steps=st.integers(min_value=1),
        max_eval_steps=st.integers(min_value=1),
    )
    def test_selects_max_steps_by_mode(
        self,
        matching_steps: int,
        mode: ExperimentMode,
        max_train_steps: int,
        max_eval_steps: int,
    ) -> None:
        max_steps = max_train_steps if mode is ExperimentMode.TRAIN else max_eval_steps
        model = _model_with_recognition(
            is_done=False, is_exploring=False, matching_steps=matching_steps
        )
        policy = MaximumSteps(max_train_steps, max_eval_steps)
        count = RecognitionCounter(0, mode)
        result = policy(model, count)
        is_done = matching_steps >= max_steps
        self.assertEqual(result.is_done, is_done)


class MinimumLMsTest(unittest.TestCase):
    @given(min_lms=st.integers(max_value=0))
    def test_raises_value_error_if_min_lms_is_not_positive(self, min_lms: int) -> None:
        with self.assertRaises(ValueError):
            MinimumLMs(min_lms)

    @given(
        conclusions=st.lists(
            st.one_of(st.none(), st.sampled_from(RecognitionConclusion)),
            min_size=1,
            max_size=20,
        ),
        min_lms=st.integers(min_value=1, max_value=20),
    )
    def test_done_iff_conclusion_count_reaches_count(
        self, conclusions: list[RecognitionConclusion | None], min_lms: int
    ) -> None:
        num_pending = conclusions.count(None)
        num_concluded = len(conclusions) - num_pending
        model = _model_with_conclusions(conclusions)
        policy = MinimumLMs(min_lms)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, num_concluded >= min_lms)


class NaiveScanTest(unittest.TestCase):
    @given(fixed_amount=st.integers(max_value=0))
    def test_raises_value_error_if_fixed_amount_is_not_positive(
        self, fixed_amount: int
    ) -> None:
        with self.assertRaises(ValueError):
            NaiveScan(fixed_amount)

    @given(step=st.integers(min_value=0))
    def test_fixed_amount_5_yields_307_steps(self, step: int) -> None:
        model = _model_is_done(is_done=False)
        policy = NaiveScan(fixed_amount=5)
        count = RecognitionCounter(step)
        result = policy(model, count)
        is_done = step >= 307
        self.assertEqual(result.is_done, is_done)

    @given(fixed_amount=st.integers(min_value=1, max_value=100))
    def test_step_limit_less_than_10000(self, fixed_amount: int) -> None:
        model = _model_is_done(is_done=False)
        policy = NaiveScan(fixed_amount)
        count = RecognitionCounter(step=10000)
        result = policy(model, count)
        self.assertTrue(result.is_done)


class ObjectRecognitionTest(unittest.TestCase):
    @given(max_total_steps=st.integers(max_value=0))
    def test_raises_value_error_if_max_total_steps_is_not_positive(
        self, max_total_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            ObjectRecognition(1, 1, max_total_steps)

    @given(
        max_train_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_train_steps_is_not_positive(
        self, max_train_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            ObjectRecognition(max_train_steps, 1, 1)

    @given(
        max_eval_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_eval_steps_is_not_positive(
        self, max_eval_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            ObjectRecognition(1, max_eval_steps, 1)

    @given(
        matching_steps=st.integers(min_value=0),
        max_steps=st.integers(min_value=1),
        step_asc=ascending_ints(min_value=0),
    )
    def test_times_out_at_or_after_max_matching_steps(
        self,
        matching_steps: int,
        max_steps: int,
        step_asc: tuple[int, int],
    ) -> None:
        (step, max_total_steps) = step_asc
        model = _model_with_recognition(
            is_done=False, is_exploring=False, matching_steps=matching_steps
        )
        policy = ObjectRecognition(max_steps, max_steps, max_total_steps)
        count = RecognitionCounter(step)
        result = policy(model, count)
        is_done = matching_steps >= max_steps
        self.assertEqual(result.is_done, is_done)
        if is_done:
            model.deal_with_time_out.assert_called_once()

    @given(
        mode=st.sampled_from(ExperimentMode),
        max_train_steps=st.integers(min_value=1),
        max_eval_steps=st.integers(min_value=1),
    )
    def test_selects_max_matching_steps_by_mode(
        self, mode: ExperimentMode, max_train_steps: int, max_eval_steps: int
    ) -> None:
        max_steps = max_train_steps if mode is ExperimentMode.TRAIN else max_eval_steps
        policy = ObjectRecognition(max_train_steps, max_eval_steps, 5000)
        at_limit = _model_with_recognition(
            is_done=False, is_exploring=False, matching_steps=max_steps
        )
        self.assertTrue(policy(at_limit, RecognitionCounter(0, mode)).is_done)
        before_limit = _model_with_recognition(
            is_done=False, is_exploring=False, matching_steps=max_steps - 1
        )
        self.assertFalse(policy(before_limit, RecognitionCounter(0, mode)).is_done)

    @given(
        max_total_steps=st.integers(min_value=1),
        extra=st.integers(min_value=0),
        is_exploring=st.booleans(),
        matching_steps=st.integers(min_value=0, max_value=100),
    )
    def test_times_out_at_or_after_max_total_steps(
        self,
        max_total_steps: int,
        extra: int,
        is_exploring: bool,
        matching_steps: int,
    ) -> None:
        max_steps = 1000
        model = _model_with_recognition(
            is_done=False, is_exploring=is_exploring, matching_steps=matching_steps
        )
        policy = ObjectRecognition(max_steps, max_steps, max_total_steps)
        count = RecognitionCounter(max_total_steps + extra)
        result = policy(model, count)
        self.assertTrue(result.is_done)
        model.deal_with_time_out.assert_called_once()

    @given(
        is_done=st.booleans(),
        is_exploring=st.booleans(),
        matching_steps=st.integers(min_value=0, max_value=100),
        step=st.integers(min_value=0, max_value=500),
    )
    def test_defers_to_model_before_max_total_steps(
        self,
        is_done: bool,
        is_exploring: bool,
        matching_steps: int,
        step: int,
    ) -> None:
        max_steps = 1000
        max_total_steps = 5000
        model = _model_with_recognition(
            is_done=is_done, is_exploring=is_exploring, matching_steps=matching_steps
        )
        policy = ObjectRecognition(max_steps, max_steps, max_total_steps)
        count = RecognitionCounter(step)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)
        model.deal_with_time_out.assert_not_called()


@st.composite
def policies_with_done_index(draw: st.DrawFn) -> tuple[list[MagicMock], int | None]:
    num_policies = draw(st.integers(min_value=1, max_value=10))
    done_index = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=num_policies - 1))
    )
    policies = [
        MagicMock(
            RecognitionPolicy, return_value=RecognitionResult(is_done=i == done_index)
        )
        for i in range(num_policies)
    ]
    return (policies, done_index)


class AnyPolicyTest(unittest.TestCase):
    def test_raises_value_error_if_policies_are_empty(self) -> None:
        with self.assertRaises(ValueError):
            AnyPolicy([])

    @given(policies_and_done_index=policies_with_done_index())
    def test_stops_at_first_policy_that_is_done(
        self, policies_and_done_index: tuple[list[MagicMock], int | None]
    ) -> None:
        (policies, done_index) = policies_and_done_index
        model = MagicMock()
        policy = AnyPolicy(policies)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, done_index is not None)
        num_called = len(policies) if done_index is None else done_index + 1
        for called in policies[:num_called]:
            called.assert_called_once()
        for not_called in policies[num_called:]:
            not_called.assert_not_called()
        model.assert_not_called()
