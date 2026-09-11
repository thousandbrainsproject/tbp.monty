# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Protocol, Sequence

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.memento import Memento


class AttentionSystemProtocol(Protocol):
    def step(
        self, goals: Sequence[Goal], regions: Sequence[AttentionRegion]
    ) -> list[Goal]: ...

    def reset(self) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopAttentionSystem(AttentionSystemProtocol):
    def step(
        self,
        goals: Sequence[Goal],
        regions: Sequence[AttentionRegion],  # noqa: ARG002
    ) -> list[Goal]:
        return list(goals)

    def reset(self) -> None:
        """Nothing to reset."""

    def state_dict(self) -> Memento:
        return {}
