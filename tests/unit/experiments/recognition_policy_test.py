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

    @given(max_steps=st.integers(min_value=1), extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, max_steps: int, extra: int) -> None:
        model = _model_is_done(is_done=False)
        policy = MaximumSteps(max_steps, max_steps)
        count = RecognitionCounter(step=max_steps + extra)
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(
        mode=st.sampled_from(ExperimentMode),
        max_train_steps=st.integers(min_value=1),
        max_eval_steps=st.integers(min_value=1),
    )
    def test_selects_max_steps_by_mode(
        self, mode: ExperimentMode, max_train_steps: int, max_eval_steps: int
    ) -> None:
        max_steps = max_train_steps if mode is ExperimentMode.TRAIN else max_eval_steps
        model = _model_is_done(is_done=False)
        policy = MaximumSteps(max_train_steps, max_eval_steps)
        at_limit = policy(model, RecognitionCounter(max_steps, mode))
        self.assertTrue(at_limit.is_done)
        before_limit = policy(model, RecognitionCounter(max_steps - 1, mode))
        self.assertFalse(before_limit.is_done)

    @given(
        is_done=st.booleans(),
        asc=ascending_ints(min_value=0),
    )
    def test_defers_to_model_before_max_steps(
        self, is_done: bool, asc: tuple[int, int]
    ) -> None:
        (step, max_steps) = asc
        model = _model_is_done(is_done)
        policy = MaximumSteps(max_steps, max_steps)
        count = RecognitionCounter(step=step)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


class MinimumCountTest(unittest.TestCase):
    @given(min_lms=st.integers(max_value=0))
    def test_raises_value_error_if_min_lms_is_not_positive(self, min_lms: int) -> None:
        with self.assertRaises(ValueError):
            MinimumLMs(min_lms, 1, 1)

    @given(
        max_train_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_train_steps_is_not_positive(
        self, max_train_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MinimumLMs(1, max_train_steps, 1)

    @given(
        max_eval_steps=st.integers(max_value=0),
    )
    def test_raises_value_error_if_max_eval_steps_is_not_positive(
        self, max_eval_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MinimumLMs(1, 1, max_eval_steps)

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
        policy = MinimumLMs(min_lms, 1, 1)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, num_concluded >= min_lms)

    def test_counts_any_conclusion_not_just_match(self) -> None:
        model = _model_with_conclusions(
            [RecognitionConclusion.NO_MATCH, RecognitionConclusion.TIME_OUT]
        )
        policy = MinimumLMs(2, 1, 1)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(max_steps=st.integers(min_value=1), extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_steps(self, max_steps: int, extra: int) -> None:
        model = _model_with_conclusions([None, None])
        policy = MinimumLMs(1, max_steps, max_steps)
        count = RecognitionCounter(max_steps + extra)
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(
        mode=st.sampled_from(ExperimentMode),
        max_train_steps=st.integers(min_value=1),
        max_eval_steps=st.integers(min_value=1),
    )
    def test_selects_max_steps_by_mode(
        self, mode: ExperimentMode, max_train_steps: int, max_eval_steps: int
    ) -> None:
        max_steps = max_train_steps if mode is ExperimentMode.TRAIN else max_eval_steps
        model = _model_with_conclusions([None, None])
        policy = MinimumLMs(1, max_train_steps, max_eval_steps)
        at_limit = policy(model, RecognitionCounter(max_steps, mode))
        self.assertTrue(at_limit.is_done)
        before_limit = policy(model, RecognitionCounter(max_steps - 1, mode))
        self.assertFalse(before_limit.is_done)


class MaxTotalStepsTest(unittest.TestCase):
    @given(max_total_steps=st.integers(max_value=0))
    def test_raises_value_error_if_max_total_steps_is_not_positive(
        self, max_total_steps: int
    ) -> None:
        with self.assertRaises(ValueError):
            MaxTotalSteps(max_total_steps)

    @given(max_total_steps=st.integers(min_value=1), extra=st.integers(min_value=0))
    def test_times_out_at_or_after_max_total_steps(
        self, max_total_steps: int, extra: int
    ) -> None:
        model = _model_is_done(is_done=False)
        policy = MaxTotalSteps(max_total_steps=max_total_steps)
        count = RecognitionCounter(step=max_total_steps + extra)
        result = policy(model, count)
        self.assertTrue(result.is_done)

    @given(
        is_done=st.booleans(),
        asc=ascending_ints(min_value=0),
    )
    def test_defers_to_model_before_max_total_steps(
        self, is_done: bool, asc: tuple[int, int]
    ) -> None:
        (step, max_total_steps) = asc
        model = _model_is_done(is_done)
        policy = MaxTotalSteps(max_total_steps=max_total_steps)
        count = RecognitionCounter(step)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


class NaiveScanTest(unittest.TestCase):
    @given(
        max_total_steps=st.integers(max_value=0), fixed_amount=st.integers(min_value=1)
    )
    def test_raises_value_error_if_max_total_steps_is_not_positive(
        self, max_total_steps: int, fixed_amount: int
    ) -> None:
        with self.assertRaises(ValueError):
            NaiveScan(max_total_steps, fixed_amount=fixed_amount)

    @given(
        max_total_steps=st.integers(min_value=1), fixed_amount=st.integers(max_value=0)
    )
    def test_raises_value_error_if_fixed_amount_is_not_positive(
        self, max_total_steps: int, fixed_amount: int
    ) -> None:
        with self.assertRaises(ValueError):
            NaiveScan(max_total_steps, fixed_amount=fixed_amount)

    @given(step=st.integers(min_value=0))
    def test_fixed_amount_5_yields_307_steps(self, step: int) -> None:
        model = _model_is_done(is_done=False)
        policy = NaiveScan(max_total_steps=500, fixed_amount=5)
        count = RecognitionCounter(step)
        result = policy(model, count)
        is_done = step >= 307
        self.assertEqual(result.is_done, is_done)

    @given(is_done=st.booleans(), step=st.integers(min_value=0, max_value=306))
    def test_defers_to_model_before_step_limit(self, is_done: bool, step: int) -> None:
        model = _model_is_done(is_done)
        policy = NaiveScan(max_total_steps=500, fixed_amount=5)
        count = RecognitionCounter(step)
        result = policy(model, count)
        self.assertEqual(result.is_done, is_done)


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
        is_done=st.booleans(),
        max_steps=st.integers(min_value=1, max_value=500),
        extra=st.integers(min_value=0, max_value=500),
        step=st.integers(min_value=0, max_value=2000),
    )
    def test_times_out_at_or_after_max_matching_steps(
        self,
        is_done: bool,
        max_steps: int,
        extra: int,
        step: int,
    ) -> None:
        max_total_steps = 5000
        matching_steps = max_steps + extra
        model = _model_with_recognition(
            is_done=is_done, is_exploring=False, matching_steps=matching_steps
        )
        policy = ObjectRecognition(max_steps, max_steps, max_total_steps)
        count = RecognitionCounter(step)
        result = policy(model, count)
        self.assertTrue(result.is_done)
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


def _mock_3_policies(first: bool, middle: bool, last: bool):
    return [
        MagicMock(RecognitionPolicy, return_value=RecognitionResult(is_done=first)),
        MagicMock(RecognitionPolicy, return_value=RecognitionResult(is_done=middle)),
        MagicMock(RecognitionPolicy, return_value=RecognitionResult(is_done=last)),
    ]


class AnyPolicyTest(unittest.TestCase):
    def test_raises_value_error_if_policies_are_empty(self) -> None:
        with self.assertRaises(ValueError):
            AnyPolicy([])

    def test_first_policy_is_done(self) -> None:
        policies = _mock_3_policies(first=True, middle=True, last=True)
        model = _model_is_done(is_done=True)
        policy = AnyPolicy(policies)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, True)
        policies[0].assert_called_once()
        policies[1].assert_not_called()
        policies[2].assert_not_called()
        model.assert_not_called()

    def test_second_policy_is_done(self) -> None:
        policies = _mock_3_policies(first=False, middle=True, last=True)
        model = _model_is_done(is_done=True)
        policy = AnyPolicy(policies)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, True)
        policies[0].assert_called_once()
        policies[1].assert_called_once()
        policies[2].assert_not_called()
        model.assert_not_called()

    def test_third_policy_is_done(self) -> None:
        policies = _mock_3_policies(first=False, middle=False, last=True)
        model = _model_is_done(is_done=True)
        policy = AnyPolicy(policies)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, True)
        policies[0].assert_called_once()
        policies[1].assert_called_once()
        policies[2].assert_called_once()
        model.assert_not_called()

    def test_no_policy_is_done(self) -> None:
        policies = _mock_3_policies(first=False, middle=False, last=False)
        model = MagicMock()
        policy = AnyPolicy(policies)
        count = RecognitionCounter()
        result = policy(model, count)
        self.assertEqual(result.is_done, False)
        policies[0].assert_called_once()
        policies[1].assert_called_once()
        policies[2].assert_called_once()
        model.assert_not_called()
