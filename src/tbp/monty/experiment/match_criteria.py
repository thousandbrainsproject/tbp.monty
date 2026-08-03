# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Mapping, Protocol

from typing_extensions import Self

__all__ = ["AllLMsMatch", "AnyLMsMatch", "MatchCriterion", "NamedLMsMatch"]


class MatchCriterion(Protocol):
    """Decides whether learning modules have collectively matched the target.

    An individual learning module reaches its own terminal state independently of the
    others. The match criterion turns those per-LM terminal states into the single
    system-level decision of whether Monty has recognized the object.
    """

    def __call__(self: Self, terminal_states: Mapping[str, str | None]) -> bool:
        """Evaluate the criterion against the given terminal states.

        Args:
            terminal_states: A mapping of learning module names to their terminal
                states. A value of `None` indicates that the learning module has not
                reached its terminal state yet.

        Returns:
            True if the criterion is met, False otherwise.
        """
        ...


class AnyLMsMatch(MatchCriterion):
    """Satisifed once any `count` of learning modules have reached "match"."""

    _count: int

    def __init__(self: Self, count: int) -> None:
        """Initialize the criterion.

        Args:
            count: The number of learning modules that must reach "match" for the
                criterion to be satisfied.

        Raises:
            ValueError: If `count` is not positive.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        self._count = count

    def __call__(self: Self, terminal_states: Mapping[str, str | None]) -> bool:
        matched = sum(1 for state in terminal_states.values() if state == "match")
        return matched >= self._count


class NamedLMsMatch(MatchCriterion):
    """Satisifed once all learning modules with the given IDs have reached "match"."""

    _ids: frozenset[str]

    def __init__(self: Self, ids: list[str]) -> None:
        """Initialize the criterion.

        Args:
            ids: The IDs of the learning modules that must reach "match" for the
                criterion to be satisfied.

        Raises:
            ValueError: If `ids` is empty.
        """
        if not ids:
            raise ValueError("ids must not be empty")
        self._ids = frozenset(ids)

    def __call__(self: Self, terminal_states: Mapping[str, str | None]) -> bool:
        unknown_ids = self._ids - terminal_states.keys()
        if unknown_ids:
            raise ValueError(f"unknown learning module IDs: {unknown_ids}")
        return all(terminal_states[lm_id] == "match" for lm_id in self._ids)


class AllLMsMatch(MatchCriterion):
    """Satisifed once all learning modules have reached "match"."""

    def __call__(self: Self, terminal_states: Mapping[str, str | None]) -> bool:
        return all(state == "match" for state in terminal_states.values())
