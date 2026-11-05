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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from tbp.monty import telemetry
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.telemetry.schemas import TelemetryEvent

__all__ = [
    "AnyStrategy",
    "MaxTotalSteps",
    "MaximumSteps",
    "MinEvalMaxTotalSteps",
    "MinimumLMs",
    "MontyIsDone",
    "NaiveScan",
    "ObjectRecognition",
    "RecognitionContext",
    "RecognitionResult",
    "RecognitionStrategy",
    "StepLimit",
]

logger = logging.getLogger(__name__)
telemeter = telemetry.getTelemeter(__name__)


@dataclass
class RecognitionContext:
    """Experiment counters and limits."""

    step: int = 0
    """The current step number."""

    matching_steps: int = 0
    """Count of matching steps taken."""

    exploring_steps: int = 0
    """Count of exploring steps taken."""

    mode: ExperimentMode = ExperimentMode.EVAL
    """The stepping mode (training or evaluation)."""


@dataclass
class RecognitionResult:
    """Aggregated result from the RecognitionStrategy."""

    is_done: bool = False
    """A terminal condition has been reached."""

    time_out_lms: bool = False
    """Request to time-out LMs that are not done yet."""

    start_exploring: bool = False
    """Request for transition to 'exploring' mode."""


class RecognitionStrategy(Protocol):
    """Decides what constitutes "recognition" in an Experiment.

    Each Learning Module determines its own Recognition Status independently of the
    others. The RecognitionStrategy turns the per-LM status into the single decision
    of whether Monty has recognized the object.
    """

    def __call__(
        self: Self, ctx: RecognitionContext, model: MontyBase
    ) -> RecognitionResult:
        """Apply this strategy to produce a Recognition Result from per-LM status.

        Args:
            ctx: The Experiment counters and limits.
            model: The Monty model to be queried.

        Returns:
            An aggregate Recognition Result based on this strategy.
        """
        ...


class MontyIsDone(RecognitionStrategy):
    """Legacy (default) recognition strategy."""

    def __call__(
        self: Self,
        ctx: RecognitionContext,  # noqa: ARG002
        model: MontyBase,
    ) -> RecognitionResult:
        is_done = model.is_done
        if is_done:
            logger.info("MontyIsDone is done, with model.is_done")
        return RecognitionResult(is_done)


class MinEvalMaxTotalSteps(RecognitionStrategy):
    """Recognition strategy that is not done while `matching_steps < min_eval_steps`.

    The primary purpose of this strategy is to return `is_done = False`
    until at least `min_eval_steps` have been taken. It includes
    a `step >= max_total_steps` check to return `is_done = True` when
    `matching_steps` fails to increment sufficiently over `max_total_steps` steps.
    """

    _wrapped: RecognitionStrategy
    """The strategy _wrapped_ and possibly overridden by this strategy."""

    _min_eval_steps: int
    """The minimum steps to take in matching mode."""

    _max_total_steps: int
    """The maximum number of steps before forcing termination."""

    def __init__(
        self: Self,
        wrapped: RecognitionStrategy,
        min_eval_steps: int = 0,
        max_total_steps: int = 1,
    ) -> None:
        """Initialize the strategy.

        Args:
            wrapped: The strategy _wrapped_ and possibly overridden by this strategy.
            min_eval_steps: The minimum steps to take in matching mode.
            max_total_steps: The maximum number of steps before forcing termination.

        Raises:
            ValueError:
                - If `min_eval_steps` is negative.
                - If `max_total_steps` is not positive.
                - If `min_eval_steps > max_total_steps`.
        """
        if min_eval_steps < 0:
            raise ValueError("min_eval_steps must be non-negative")
        if max_total_steps <= 0:
            raise ValueError("max_total_steps must be positive")
        if min_eval_steps > max_total_steps:
            raise ValueError(
                "min_eval_steps must be less than or equal to max_total_steps"
            )

        self._wrapped = wrapped
        self._min_eval_steps = min_eval_steps
        self._max_total_steps = max_total_steps

    def __call__(
        self: Self,
        ctx: RecognitionContext,
        model: MontyBase,
    ) -> RecognitionResult:
        wrapped = self._wrapped(ctx, model)
        if model.is_done:
            result = RecognitionResult(
                is_done=True,
                time_out_lms=wrapped.time_out_lms,
                start_exploring=wrapped.start_exploring,
            )
        elif ctx.step >= self._max_total_steps:
            result = RecognitionResult(
                is_done=True,
                time_out_lms=True,
                start_exploring=wrapped.start_exploring,
            )
        elif (
            ctx.mode is ExperimentMode.EVAL
            and ctx.matching_steps < self._min_eval_steps
        ):
            result = RecognitionResult(
                is_done=False,
                time_out_lms=wrapped.time_out_lms,
                start_exploring=wrapped.start_exploring,
            )
        else:
            result = wrapped

        if result.is_done:
            logger.info(
                "MinEvalMaxTotalSteps is done, "
                "with model.is_done=%s step=%d matching_steps=%d",
                model.is_done,
                ctx.step,
                ctx.matching_steps,
            )
            lms = {
                str(lm.learning_module_id): lm.recognition_status
                for lm in model.learning_modules
            }
            telemeter.info(
                TelemetryEvent(
                    kind="TerminalCondition",
                    model=dict(
                        is_done=model.is_done,
                        is_exploring=model.is_exploring,
                        lms=lms,
                    ),
                    ctx=ctx,
                    result=result,
                )
            )

        return result


class MaxTotalSteps(RecognitionStrategy):
    """`step >= max_total_steps`."""

    _max_total_steps: int
    """The maximum number of steps before terminating the episode."""

    def __init__(self: Self, max_total_steps: int) -> None:
        """Initialize the strategy.

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
        ctx: RecognitionContext,
        model: MontyBase,  # noqa: ARG002
    ) -> RecognitionResult:
        is_done = ctx.step >= self._max_total_steps
        if is_done:
            logger.info("MaxTotalSteps is done, with step=%d", ctx.step)
        return RecognitionResult(is_done=is_done, time_out_lms=is_done)


class MaximumSteps(RecognitionStrategy):
    """`model.matching_steps >= {max_train_steps | max_eval_steps}`."""

    _max_train_steps: int
    """The maximum steps to take in training mode."""

    _max_eval_steps: int
    """The maximum steps to take in evaluation mode."""

    def __init__(self: Self, max_train_steps: int, max_eval_steps: int) -> None:
        """Initialize the strategy.

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
        ctx: RecognitionContext,
        model: MontyBase,
    ) -> RecognitionResult:
        max_steps = (
            self._max_train_steps
            if ctx.mode is ExperimentMode.TRAIN
            else self._max_eval_steps
        )
        is_done = (not model.is_exploring) and (model.matching_steps >= max_steps)
        result = RecognitionResult(
            is_done=is_done,
            time_out_lms=is_done,
        )
        if result.is_done:
            logger.info(
                "MaximumSteps is done, "
                "with model.is_exploring=%s model.matching_steps=%d",
                model.is_exploring,
                model.matching_steps,
            )
        return result


class StepLimit(RecognitionStrategy):
    """Recognition strategy that checks step counter limits.

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
        """Initialize the strategy.

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
        self: Self, ctx: RecognitionContext, model: MontyBase
    ) -> RecognitionResult:
        if model.is_exploring:
            is_done = ctx.exploring_steps >= self._num_exploring_steps
            result = RecognitionResult(
                is_done=is_done,
                time_out_lms=is_done,
            )
        elif ctx.mode is ExperimentMode.TRAIN:
            is_done = ctx.matching_steps >= self._max_train_steps
            result = RecognitionResult(
                is_done=is_done,
                time_out_lms=is_done,
                start_exploring=(ctx.matching_steps >= self._min_train_steps),
            )
        elif ctx.mode is ExperimentMode.EVAL:
            is_done = ctx.matching_steps >= self._max_eval_steps
            result = RecognitionResult(
                is_done=is_done,
                time_out_lms=is_done,
            )
        else:
            result = RecognitionResult()

        if result.is_done:
            logger.info(
                "StepLimit is done, with matching_steps=%d exploring_steps=%d",
                ctx.matching_steps,
                ctx.exploring_steps,
            )

        return result


class MatchCriteria(RecognitionStrategy):
    """The `model.learning_modules` meet the configured `match_criteria`."""

    def __call__(
        self: Self,
        ctx: RecognitionContext,  # noqa: ARG002
        model: MontyBase,
    ) -> RecognitionResult:
        terminal_states: dict[str, str | None] = {
            lm.learning_module_id: lm.terminal_state for lm in model.learning_modules
        }
        # TODO: Either configure the `match_criterion` on the strategy itself,
        #       or provide configurable strategies for each criterion.
        is_done = model._match_criterion(terminal_states)
        if is_done:
            logger.info(
                "MatchCriteria is done, with terminal_states=%s",
                terminal_states,
            )
        return RecognitionResult(is_done=is_done, time_out_lms=is_done)


class MinimumLMs(RecognitionStrategy):
    """`min_lms` have reached a conclusion."""

    _min_lms: int
    """The minimum number of LMs that must reach a conclusion."""

    def __init__(self: Self, min_lms: int) -> None:
        """Initialize the strategy.

        Args:
            min_lms: The number of Learning Modules that must reach a conclusion for
                the strategy to be satisfied.

        Raises:
            ValueError: If `min_lms` is not positive.
        """
        if min_lms <= 0:
            raise ValueError("min_lms must be positive")

        self._min_lms = min_lms

    def __call__(
        self: Self,
        ctx: RecognitionContext,  # noqa: ARG002
        model: MontyBase,
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


class NaiveScan(RecognitionStrategy):
    """`steps >= step_limit`, with `step_limit` derived from `fixed_amount`.

    The `step_limit` is the number of steps the Naive Scan motor policy takes
    before its spiral completes.
    """

    _step_limit: int
    """The maximum number of steps before terminating the episode."""

    def __init__(self: Self, fixed_amount: int) -> None:
        """Initialize the strategy.

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
        ctx: RecognitionContext,
        model: MontyBase,  # noqa: ARG002
    ) -> RecognitionResult:
        is_done = ctx.step >= self._step_limit
        if is_done:
            logger.info("NaiveScan is done, with step=%d", ctx.step)
        return RecognitionResult(is_done)


class ObjectRecognition(RecognitionStrategy):
    """Determine terminal conditions for object recognition experiments.

    Terminal conditions include:
    - `model.matching_steps >= {max_train_steps | max_eval_steps}`
    - `ctx.step >= max_total_steps`
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
        """Initialize the strategy.

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
        self: Self, ctx: RecognitionContext, model: MontyBase
    ) -> RecognitionResult:
        max_steps = (
            self._max_train_steps
            if ctx.mode is ExperimentMode.TRAIN
            else self._max_eval_steps
        )
        if (not model.is_exploring) and (model.matching_steps >= max_steps):
            result = RecognitionResult(is_done=True, time_out_lms=True)
            logger.info(
                "ObjectRecognition is done, "
                "with model.is_exploring=%s model.matching_steps=%d",
                model.is_exploring,
                model.matching_steps,
            )
        elif ctx.step >= self._max_total_steps:
            result = RecognitionResult(is_done=True, time_out_lms=True)
            logger.info("ObjectRecognition is done, with step=%d", ctx.step)
        elif model.is_done:
            result = RecognitionResult(is_done=True)
            logger.info("ObjectRecognition is done, with model.is_done")
        else:
            result = RecognitionResult()
        return result


class AnyStrategy(RecognitionStrategy):
    """Combine multiple terminal conditions for Experiments.

    Terminal condition is reached if _any_ `RecognitionStrategy` says so.
    """

    _strategies: Sequence[RecognitionStrategy]
    """The strategies to check (in order)."""

    def __init__(self: Self, strategies: Sequence[RecognitionStrategy]) -> None:
        """Initialize the strategy.

        Args:
            strategies: The strategies to check (in order).

        Raises:
            ValueError: If `len(strategies) < 1`.
        """
        if len(strategies) < 1:
            raise ValueError("no strategies to check")

        self._strategies = strategies

    def __call__(
        self: Self, ctx: RecognitionContext, model: MontyBase
    ) -> RecognitionResult:
        result = RecognitionResult()
        for strategy in self._strategies:
            rr = strategy(ctx, model)
            result = RecognitionResult(
                is_done=result.is_done or rr.is_done,
                time_out_lms=result.time_out_lms or rr.time_out_lms,
                start_exploring=result.start_exploring or rr.start_exploring,
            )
            if result.is_done:
                break
        if result.is_done:
            logger.info("AnyStrategy is done, with %s", result)
        return result
