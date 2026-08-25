# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from typing_extensions import Self

__all__ = [
    "MinimumCount",
    "RecognitionConclusion",
    "RecognitionPolicy",
    "RecognitionResult",
    "RecognitionStatus",
]


class RecognitionConclusion(Enum):
    """Label for the terminal state of a Learning Module."""

    MATCH = "match"
    NO_MATCH = "no_match"
    TIME_OUT = "time_out"


@dataclass
class RecognitionStatus:
    """Recognition Status from each Learning Module."""

    conclusion: RecognitionConclusion | None
    telemetry: dict[str, Any] = field(default_factory=dict)


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
        self: Self, step: int, status: Mapping[str, RecognitionStatus]
    ) -> RecognitionResult:
        """Apply this policy to produce a Recognition Result from per-LM status.

        Args:
            step: The experiment step number.
            status: A mapping of Learning Module names to their Recognition Status.

        Returns:
            An aggregate Recognition Result based on this policy.
        """
        ...


class MinimumCount(RecognitionPolicy):
    """Satisfied once any `count` of Learning Modules have reached "match"."""

    _max_steps: int
    """The maximum number of Monty steps before reaching a conclusion."""

    _count: int
    """The minimum number of LMs that must reach "match" status."""

    def __init__(self: Self, count: int, max_steps: int) -> None:
        """Initialize the policy.

        Args:
            count: The number of Learning Modules that must reach "match" for the
                policy to be satisfied.
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

    def __call__(
        self: Self, step: int, status: Mapping[str, RecognitionStatus]
    ) -> RecognitionResult:
        if step >= self._max_steps:
            return RecognitionResult(is_done=True)

        num_matched = sum(1 for rs in status.values() if rs.conclusion is not None)
        is_done = num_matched >= self._count
        return RecognitionResult(is_done=is_done)
