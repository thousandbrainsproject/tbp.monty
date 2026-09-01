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

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.models.monty_base import MontyBase

__all__ = [
    "MinimumCount",
    "MontyIsDone",
    "RecognitionPolicy",
    "RecognitionResult",
]


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
        self: Self, exp: MontyExperiment, model: MontyBase, step: int
    ) -> RecognitionResult:
        """Apply this policy to produce a Recognition Result from per-LM status.

        Args:
            exp: The Experiment to be queried.
            model: The Monty model to be queried.
            step: The Experiment step number.

        Returns:
            An aggregate Recognition Result based on this policy.
        """
        ...


class MontyIsDone(RecognitionPolicy):
    """Legacy (default) policy."""

    def __call__(
        self: Self, exp: MontyExperiment, model: MontyBase, step: int
    ) -> RecognitionResult:
        if step >= exp.max_steps:
            return RecognitionResult(is_done=True)

        return RecognitionResult(is_done=model.is_done)


class MinimumCount(RecognitionPolicy):
    """`count` LMs have reached a conclusion"""

    _count: int
    """The minimum number of LMs that must reach a conclusion."""

    def __init__(self: Self, count: int) -> None:
        """Initialize the policy.

        Args:
            count: The number of Learning Modules that must reach a conclusion for
                the policy to be satisfied.

        Raises:
            ValueError: If `count` is not positive.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        self._count = count

    def __call__(
        self: Self, exp: MontyExperiment, model: MontyBase, step: int
    ) -> RecognitionResult:
        if step >= exp.max_steps:
            return RecognitionResult(is_done=True)

        num_matched = sum(
            1
            for lm in model.learning_modules
            if lm.recognition_status.conclusion is not None
        )
        is_done = num_matched >= self._count

        return RecognitionResult(is_done=is_done)
