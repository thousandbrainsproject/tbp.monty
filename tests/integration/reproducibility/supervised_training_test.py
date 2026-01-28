# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import tempfile
import unittest
from pathlib import Path

import hydra
import torch

from tests.integration.reproducibility.asserts import assert_trained_models_equal
from tests.integration.reproducibility.config import hydra_config
from tests.integration.reproducibility.run import parallel_run, serial_run


class SupervisedTrainingTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(tempfile.mkdtemp())

    def test_training_results_are_equal(self):
        with hydra.initialize(version_base=None, config_path="../../../conf"):
            config = hydra_config(
                "reproducibility_supervised_training",
                self.output_dir,
                # Note: Since training episodes are not reproducible between run.py and
                #       run_parallel.py, we must use the same fixed actions for both the
                #       serial and parallel runs to get the same training results.
                fixed_actions_path=(
                    Path(__file__).parent / "supervised_training_actions.jsonl"
                ),
            )

            serial_run(config)
            serial_model_path = (
                Path(config.experiment.config.logging.output_dir)
                / "pretrained"
                / "model.pt"
            )

            config = hydra_config(
                "reproducibility_supervised_training",
                self.output_dir,
                # Note: Since training episodes are not reproducible between run.py and
                #       run_parallel.py, we must use the same fixed actions for both the
                #       serial and parallel runs to get the same training results.
                fixed_actions_path=(
                    Path(__file__).parent / "supervised_training_actions.jsonl"
                ),
            )

            parallel_run(config)
            parallel_model_path = (
                Path(config.experiment.config.logging.output_dir)
                / config.experiment.config.logging.run_name
                / "pretrained"
                / "model.pt"
            )

        serial_model = torch.load(serial_model_path)
        parallel_model = torch.load(parallel_model_path)
        assert_trained_models_equal(serial_model, parallel_model)
