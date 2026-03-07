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
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)


class PretrainingExperimentsTest(unittest.TestCase):
    def test_pretraining_output_dir_update_works_for_struct_dictconfig(self) -> None:
        output_dir = "monty_pretraining_test"
        config = OmegaConf.create({"logging": {"output_dir": output_dir}})
        OmegaConf.set_struct(config, value=True)

        with patch.object(MontyExperiment, "__init__", return_value=None) as base_init:
            exp = MontySupervisedObjectPretrainingExperiment(config)

        self.assertEqual(
            config["logging"]["output_dir"],
            Path(output_dir) / "pretrained",
        )
        self.assertEqual(exp.first_epoch_object_location, {})
        base_init.assert_called_once_with(config)
