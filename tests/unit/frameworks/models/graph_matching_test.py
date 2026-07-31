# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching


class MontyForGraphMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        sensor_module = MagicMock()
        sensor_module.sensor_module_id = "sensor_module_0"

        self.learning_modules = []
        for lm_id in [
            "learning_module_0",
            "learning_module_1",
            "learning_module_2",
        ]:
            learning_module = MagicMock()
            learning_module.learning_module_id = lm_id
            learning_module.terminal_state = None
            self.learning_modules.append(learning_module)

        self.model = MontyForGraphMatching(
            sensor_modules=[sensor_module],
            learning_modules=self.learning_modules,
            motor_system=MagicMock(),
            sm_to_agent_dict={"sensor_module_0": "agent_id_0"},
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=None,
            min_eval_steps=0,
            min_train_steps=0,
            num_exploratory_steps=0,
            max_total_steps=100,
        )
        self.model.set_experiment_mode(ExperimentMode.EVAL)
        self.model.matching_steps = 1

    def set_terminal_states(self, *states: str | None) -> None:
        self.assertEqual(len(self.learning_modules), len(states))
        for learning_module, terminal_state in zip(self.learning_modules, states):
            learning_module.terminal_state = terminal_state

    def test_integer_requires_any_configured_number_of_matches(self) -> None:
        self.model.min_lms_match = 2
        self.set_terminal_states("match", None, None)
        self.assertFalse(self.model.check_terminal_conditions())

        self.set_terminal_states("match", None, "match")
        self.assertTrue(self.model.check_terminal_conditions())

    def test_string_requires_the_named_lm(self) -> None:
        self.model.min_lms_match = "learning_module_1"
        self.set_terminal_states("match", None, "match")
        self.assertFalse(self.model.check_terminal_conditions())

        self.set_terminal_states("match", "match", "match")
        self.assertTrue(self.model.check_terminal_conditions())

    def test_hydra_list_requires_every_named_lm(self) -> None:
        self.model.min_lms_match = OmegaConf.create(
            ["learning_module_0", "learning_module_2"]
        )
        self.set_terminal_states("match", "match", None)
        self.assertFalse(self.model.check_terminal_conditions())

        self.set_terminal_states("match", "match", "match")
        self.assertTrue(self.model.check_terminal_conditions())

    def test_rejects_empty_named_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one LM ID"):
            self.model.min_lms_match = []

    def test_rejects_unknown_lm_id(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown LM IDs.*learning_module_3.*Available LM IDs",
        ):
            self.model.min_lms_match = "learning_module_3"

    def test_rejects_non_string_lm_id(self) -> None:
        with self.assertRaisesRegex(TypeError, "Every LM ID"):
            self.model.min_lms_match = ["learning_module_0", 1]  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
