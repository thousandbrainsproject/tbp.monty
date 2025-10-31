# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.states import GoalState


class FakeLearningModule(LearningModule):
    """Dummy placeholder class used only for tests."""

    def __init__(self):
        self.test_attr_1 = True
        self.test_attr_2 = True

    def reset(self):
        pass

    def matching_step(self, inputs):
        pass

    def exploratory_step(self, inputs):
        pass

    def receive_votes(self, inputs):
        pass

    def send_out_vote(self):
        pass

    def state_dict(self):
        return dict(test_attr_1=self.test_attr_1, test_attr_2=self.test_attr_2)

    def load_state_dict(self, state_dict):
        self.test_attr_1 = state_dict["test_attr_1"]
        self.test_attr_2 = state_dict["test_attr_2"]

    def pre_episode(self):
        pass

    def post_episode(self):
        pass

    def set_experiment_mode(self, inputs):
        pass

    def propose_goal_states(self) -> list[GoalState]:
        return []

    def get_output(self):
        pass
