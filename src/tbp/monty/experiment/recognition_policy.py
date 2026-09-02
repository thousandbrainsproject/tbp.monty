# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from typing_extensions import Self

from tbp.monty.frameworks.models.monty_base import MontyBase

__all__ = [
    "MinimumLMs",
    "MontyIsDone",
    "RecognitionPolicy",
    "RecognitionResult",
]


@dataclass
class RecognitionCounter:
    """Experiment counters and limits."""

    step: int = 0
    max_steps: int = 0


@dataclass
class RecognitionResult:
    """Aggregated result from the Recognition Policy."""

    is_done: bool


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
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        if count.step >= count.max_steps:
            return RecognitionResult(is_done=True)

        return RecognitionResult(is_done=model.is_done)


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
        self: Self, model: MontyBase, count: RecognitionCounter
    ) -> RecognitionResult:
        if count.step >= count.max_steps:
            return RecognitionResult(is_done=True)

        num_matched = sum(
            1
            for lm in model.learning_modules
            if lm.recognition_status.conclusion is not None
        )
        is_done = num_matched >= self._min_lms

        return RecognitionResult(is_done=is_done)
