# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any, Collection, Sequence

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.memento import Memento

__all__ = ["FakeLearningModule"]


class FakeLearningModule(LearningModule):
    """Dummy placeholder class used only for tests."""

    def __init__(self):
        self.test_attr_1 = True
        self.test_attr_2 = True

    def reset(self):
        pass

    def matching_step(self, ctx: RuntimeContext, percepts: Sequence[Message]) -> None:
        pass

    def exploratory_step(
        self, ctx: RuntimeContext, percepts: Sequence[Message]
    ) -> None:
        pass

    def receive_votes(self, votes: Collection[Any]) -> None:
        pass

    def send_out_vote(self) -> Any:
        pass

    def state_dict(self) -> Memento:
        return dict(test_attr_1=self.test_attr_1, test_attr_2=self.test_attr_2)

    def load_state_dict(self, memento: Memento) -> None:
        self.test_attr_1 = memento["test_attr_1"]
        self.test_attr_2 = memento["test_attr_2"]

    def reset_stm(self) -> None:
        pass

    def fixme_reset_ground_truth(self, primary_target=None) -> None:
        pass

    def update_ltm_from_stm(self):
        pass

    def fixme_update_ground_truth(self):
        pass

    def set_experiment_mode(self, mode: ExperimentMode):
        pass

    def propose_goals(self) -> list[Goal]:
        return []

    def get_output(self) -> Message | None:
        pass
