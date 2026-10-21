# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from typing_extensions import Self

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.monty_base import MontyBase

__all__ = [
    "AnyPolicy",
    "MaxTotalSteps",
    "MaximumSteps",
    "MinimumLMs",
    "MontyIsDone",
    "NaiveScan",
    "ObjectRecognition",
    "RecognitionCounter",
    "RecognitionPolicy",
    "RecognitionResult",
    "StepLimit",
]

logger = logging.getLogger(__name__)


@dataclass
class RecognitionCounter:
    """Experiment counters and limits."""

    step: int = 0
    """The current step number."""

    matching_steps: int = 0
    """Count of matching steps taken."""

    exploring_steps: int = 0
    """Count of exploring steps taken."""

    mode: ExperimentMode = ExperimentMode.EVAL
    """The stepping mode (traning or evaluation)."""


@dataclass
class RecognitionResult:
    """Aggregated result from the Recognition Policy."""

    is_done: bool = False
    """A terminal condition has been reached."""

    is_time_out: bool = False
    """A time-out condition has been reached."""

    start_exploring: bool = False
    """A transition to 'exploring' mode is requested."""


class RecognitionPolicy(Protocol):
    """Decides what constitutes "recognition" in an Experiment.

    Each Learning Module determines its own Recognition Status independently of the
    others. The Recognition Policy turns the per-LM status into the single decision
    of whether Monty has recognized the object.
    """

    def __call__(
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        """Apply this policy to produce a Recognition Result from per-LM status.

        Args:
            model: The Monty model to be queried.
            count: The Experiment counters and limits.

        Returns:
            An aggregate Recognition Result based on this policy.
        """
        ...


class MontyIsDone(RecognitionPolicy):
    """Legacy (default) policy."""

    def __call__(
        self: Self,
        model: MontyBase,
        count: RecognitionCounter,  # noqa: ARG002
    ) -> RecognitionResult:
        is_done = model.is_done
        if is_done:
            logger.info("MontyIsDone is done, with model.is_done")
        return RecognitionResult(is_done)


class MaxTotalSteps(RecognitionPolicy):
    """`step >= max_total_steps`."""

    _max_total_steps: int
    """The maximum number of steps before terminating the episode."""

    def __init__(self: Self, max_total_steps: int) -> None:
        """Initialize the policy.

        Args:
            max_total_steps: The maximum number of steps before terminating the episode.

        Raises:
            ValueError: If `max_total_steps` is not positive.
        """
        if max_total_steps <= 0:
            raise ValueError("max_total_steps must be positive")
        self._max_total_steps = max_total_steps

    def __call__(
        self: Self,
        model: MontyBase,  # noqa: ARG002
        count: RecognitionCounter,
    ) -> RecognitionResult:
        is_done = count.step >= self._max_total_steps
        if is_done:
            logger.info("MaxTotalSteps is done, with step=%d", count.step)
        return RecognitionResult(is_done)


class MaximumSteps(RecognitionPolicy):
    """`model.matching_steps >= {max_train_steps | max_eval_steps}`."""

    _max_train_steps: int
    """The maximum steps to take in training mode."""

    _max_eval_steps: int
    """The maximum steps to take in evaluation mode."""

    def __init__(self: Self, max_train_steps: int, max_eval_steps: int) -> None:
        """Initialize the policy.

        Args:
            max_train_steps: The maximum steps to take in training mode.
            max_eval_steps: The maximum steps to take in evaluation mode.

        Raises:
            ValueError: If `max_train_steps`, or `max_eval_steps` are not positive.
        """
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")

        if max_eval_steps <= 0:
            raise ValueError("max_eval_steps must be positive")

        self._max_train_steps = max_train_steps
        self._max_eval_steps = max_eval_steps

    def __call__(
        self: Self,
        model: MontyBase,
        count: RecognitionCounter,
    ) -> RecognitionResult:
        max_steps = (
            self._max_train_steps
            if count.mode is ExperimentMode.TRAIN
            else self._max_eval_steps
        )
        is_done = (not model.is_exploring) and (model.matching_steps >= max_steps)
        result = RecognitionResult(
            is_done=is_done,
            is_time_out=is_done,
        )
        if result.is_done:
            logger.info(
                "MaximumSteps is done, "
                "with model.is_exploring=%s model.matching_steps=%d",
                model.is_exploring,
                model.matching_steps,
            )
        return result


class StepLimit(RecognitionPolicy):
    """Check step counter limits.

    Terminal conditions include:
    - `matching_steps >= {max_train_steps | max_eval_steps}`
    - `exploring_steps >= num_exploring_steps`
    """

    _min_train_steps: int
    """The minimum steps to take in training mode."""

    _num_exploring_steps: int
    """The number of steps to take in exploring mode."""

    _max_train_steps: int
    """The maximum steps to take in training mode."""

    _max_eval_steps: int
    """The maximum steps to take in evaluation mode."""

    def __init__(
        self: Self,
        min_train_steps: int = 0,
        num_exploring_steps: int = 0,
        max_train_steps: int = 1,
        max_eval_steps: int = 1,
    ) -> None:
        """Initialize the policy.

        Args:
            min_train_steps: The minimum steps to take in training mode.
            num_exploring_steps: The number of steps to take in exploring mode.
            max_train_steps: The maximum steps to take in training mode.
            max_eval_steps: The maximum steps to take in evaluation mode.

        Raises:
            ValueError:
                - If `min_train_steps` is negative.
                - If `num_exploring_steps` is negative.
                - If `max_train_steps` is not positive.
                - If `max_eval_steps` is not positive.
        """
        if min_train_steps < 0:
            raise ValueError("min_train_steps must be non-negative")
        if num_exploring_steps < 0:
            raise ValueError("num_exploring_steps must be non-negative")
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        if max_eval_steps <= 0:
            raise ValueError("max_eval_steps must be positive")

        self._min_train_steps = min_train_steps
        self._num_exploring_steps = num_exploring_steps
        self._max_train_steps = max_train_steps
        self._max_eval_steps = max_eval_steps

    def __call__(
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        if model.is_exploring:
            is_done = count.exploring_steps >= self._num_exploring_steps
            result = RecognitionResult(
                is_done=is_done,
                is_time_out=is_done,
            )
        elif count.mode is ExperimentMode.TRAIN:
            is_done = count.matching_steps >= self._max_train_steps
            result = RecognitionResult(
                is_done=is_done,
                is_time_out=is_done,
                start_exploring=(count.matching_steps >= self._min_train_steps),
            )
        elif count.mode is ExperimentMode.EVAL:
            is_done = count.matching_steps >= self._max_eval_steps
            result = RecognitionResult(
                is_done=is_done,
                is_time_out=is_done,
            )
        else:
            result = RecognitionResult()

        if result.is_done:
            logger.info(
                "StepLimit is done, with matching_steps=%d exploring_steps=%d",
                count.matching_steps,
                count.exploring_steps,
            )

        return result


class MinimumLMs(RecognitionPolicy):
    """`min_lms` have reached a conclusion."""

    _min_lms: int
    """The minimum number of LMs that must reach a conclusion."""

    def __init__(self: Self, min_lms: int) -> None:
        """Initialize the policy.

        Args:
            min_lms: The number of Learning Modules that must reach a conclusion for
                the policy to be satisfied.

        Raises:
            ValueError: If `min_lms` is not positive.
        """
        if min_lms <= 0:
            raise ValueError("min_lms must be positive")

        self._min_lms = min_lms

    def __call__(
        self: Self,
        model: MontyBase,
        count: RecognitionCounter,  # noqa: ARG002
    ) -> RecognitionResult:
        num_matched = sum(
            1
            for lm in model.learning_modules
            if lm.recognition_status.conclusion is not None
        )
        is_done = num_matched >= self._min_lms
        if is_done:
            logger.info("MinimumLMs is done, with num_matched=%d", num_matched)
        return RecognitionResult(is_done)


class NaiveScan(RecognitionPolicy):
    """`steps >= step_limit`, with `step_limit` derived from `fixed_amount`.

    The `step_limit` is the number of steps the Naive Scan motor policy takes
    before its spiral completes.
    """

    _step_limit: int
    """The maximum number of steps before terminating the episode."""

    def __init__(self: Self, fixed_amount: int) -> None:
        """Initialize the policy.

        Args:
            fixed_amount: The Naive Scan step size.

        Raises:
            ValueError: If `fixed_amount` is not positive.
        """
        if fixed_amount <= 0:
            raise ValueError("fixed_amount must be positive")

        k = math.ceil(90 / fixed_amount)  # (>=90)->1, 10->9, 5->18, 1->90
        self._step_limit = k * (k - 1) + 1  # 1->1, 9->73, 18->307, 90->8011

    def __call__(
        self: Self,
        model: MontyBase,  # noqa: ARG002
        count: RecognitionCounter,
    ) -> RecognitionResult:
        is_done = count.step >= self._step_limit
        if is_done:
            logger.info("NaiveScan is done, with step=%d", count.step)
        return RecognitionResult(is_done)


class ObjectRecognition(RecognitionPolicy):
    """Determine terminal conditions for object recognition experiments.

    Terminal conditions include:
    - `model.matching_steps >= {max_train_steps | max_eval_steps}`
    - `count.step >= max_total_steps`
    - `model.is_done`
    """

    _max_train_steps: int
    """The maximum steps to take in training mode."""

    _max_eval_steps: int
    """The maximum steps to take in evaluation mode."""

    _max_total_steps: int
    """The maximum total number of steps before terminating."""

    def __init__(
        self: Self, max_train_steps: int, max_eval_steps: int, max_total_steps: int
    ) -> None:
        """Initialize the policy.

        Args:
            max_train_steps: The maximum steps to take in training mode.
            max_eval_steps: The maximum steps to take in evaluation mode.
            max_total_steps: The maximum total number of steps before terminating.

        Raises:
            ValueError: If `max_train_steps`, `max_eval_steps`, or `max_total_steps`
                are not positive.
        """
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")

        if max_eval_steps <= 0:
            raise ValueError("max_eval_steps must be positive")

        if max_total_steps <= 0:
            raise ValueError("max_total_steps must be positive")

        self._max_train_steps = max_train_steps
        self._max_eval_steps = max_eval_steps
        self._max_total_steps = max_total_steps

    def __call__(
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        max_steps = (
            self._max_train_steps
            if count.mode is ExperimentMode.TRAIN
            else self._max_eval_steps
        )
        if (not model.is_exploring) and (model.matching_steps >= max_steps):
            result = RecognitionResult(is_done=True, is_time_out=True)
            logger.info(
                "ObjectRecognition is done, "
                "with model.is_exploring=%s model.matching_steps=%d",
                model.is_exploring,
                model.matching_steps,
            )
        elif count.step >= self._max_total_steps:
            result = RecognitionResult(is_done=True, is_time_out=True)
            logger.info("ObjectRecognition is done, with step=%d", count.step)
        elif model.is_done:
            result = RecognitionResult(is_done=True)
            logger.info("ObjectRecognition is done, with model.is_done")
        else:
            result = RecognitionResult()
        return result


class AnyPolicy(RecognitionPolicy):
    """Combine multiple terminal conditions for Experiments.

    Terminal condition is reached if _any_ `RecognitionPolicy` says so.
    """

    _policies: Sequence[RecognitionPolicy]
    """The policies to check (in order)."""

    def __init__(self: Self, policies: Sequence[RecognitionPolicy]) -> None:
        """Initialize the policy.

        Args:
            policies: The policies to check (in order).

        Raises:
            ValueError: If `len(policies) < 1`.
        """
        if len(policies) < 1:
            raise ValueError("no policies to check")

        self._policies = policies

    def __call__(
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        result = RecognitionResult()
        for policy in self._policies:
            rr = policy(model, count)
            result = RecognitionResult(
                is_done=result.is_done or rr.is_done,
                is_time_out=result.is_time_out or rr.is_time_out,
                start_exploring=result.start_exploring or rr.start_exploring,
            )
            if result.is_done:
                break
        if result.is_done:
            logger.info("AnyPolicy is done, with %s", result)
        return result
