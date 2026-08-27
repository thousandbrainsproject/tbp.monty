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
    "MinimumCount",
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

    def __call__(self: Self, model: MontyBase, step: int) -> RecognitionResult:
        """Apply this policy to produce a Recognition Result from per-LM status.

        Args:
            model: The Monty model to be queried.
            step: The Experiment step number.

        Returns:
            An aggregate Recognition Result based on this policy.
        """
        ...


class MontyIsDone(RecognitionPolicy):
    """Monty `model.is_done == True` (legacy policy)."""

    _max_steps: int | None
    """The maximum number of Monty steps before reaching a conclusion."""

    def __init__(self: Self, max_steps: int | None = None) -> None:
        """Initialize the policy.

        Args:
            max_steps: The maximum number of Monty steps before reaching a conclusion.

        Raises:
            ValueError: If `max_steps` is not `None` and not positive.
        """
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._max_steps = max_steps

    def __call__(self: Self, model: MontyBase, step: int) -> RecognitionResult:
        if self._max_steps is not None and step >= self._max_steps:
            return RecognitionResult(is_done=True)
        return RecognitionResult(is_done=model.is_done)


class MinimumCount(RecognitionPolicy):
    """`count` LMs have reached a conclusion, or `max_steps` have been taken."""

    _count: int
    """The minimum number of LMs that must reach a conclusion."""

    _max_steps: int
    """The maximum number of Monty steps before reaching a conclusion."""

    def __init__(self: Self, count: int, max_steps: int) -> None:
        """Initialize the policy.

        Args:
            count: The number of Learning Modules that must reach a conclusion for
                the policy to be satisfied.
            max_steps: The maximum number of Monty steps before reaching a conclusion.

        Raises:
            ValueError: If `count` or `max_steps` are not positive.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        self._count = count

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._max_steps = max_steps

    def __call__(self: Self, model: MontyBase, step: int) -> RecognitionResult:
        if step >= self._max_steps:
            return RecognitionResult(is_done=True)

        num_matched = sum(
            1
            for lm in model.learning_modules
            if lm.recognition_status.conclusion is not None
        )
        is_done = num_matched >= self._count
        return RecognitionResult(is_done=is_done)
