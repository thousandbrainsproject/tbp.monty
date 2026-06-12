# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import hydra
import numpy as np

from tbp.monty.frameworks.run import main
from tests import HYDRA_ROOT

DATASET_LEN = 1000
TRAIN_EPOCHS = 2
MAX_TRAIN_STEPS = 10
MAX_EVAL_STEPS = 5
EVAL_EPOCHS = 1
FAKE_OBS = np.random.rand(DATASET_LEN, 64, 64, 1)
EXPECTED_LOG = []

# Train steps
EXPECTED_LOG += ["pre_train"]
EXPECTED_LOG += [
    "pre_epoch",
    "pre_episode",
    "post_episode",
    "post_epoch",
] * TRAIN_EPOCHS
EXPECTED_LOG += ["post_train"]

# Eval steps
EXPECTED_LOG += ["pre_eval"]
EXPECTED_LOG += [
    "pre_epoch",
    "pre_episode",
    "post_episode",
    "post_epoch",
] * EVAL_EPOCHS
EXPECTED_LOG += ["post_eval"]


class MontyRunTest(unittest.TestCase):
    """Test for the `main` function of `run.py`.

    This tests that the logic in the `main` function drives an experiment through
    the expected steps of the training and evaluation process.
    """

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/run_mujoco",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_main(self):
        main(self.cfg)

        output_dir = Path(self.cfg.experiment.config.logging.output_dir)

        with (output_dir / "fake_log.json").open("r") as f:
            exp_log = json.load(f)

        self.assertListEqual(exp_log, EXPECTED_LOG)
