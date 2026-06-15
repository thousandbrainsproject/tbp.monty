# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import shutil
import tempfile
import unittest

import hydra

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests import HYDRA_ROOT


class DetailedEvidenceLmLoggingConfigTest(unittest.TestCase):
    # TODO: This test seems to primarily test our DetailedJSONHandler, and we should
    #   do that directly instead of testing to see if the Python `logging` library
    #   did what it is supposed to do.
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/evidence_lm/base_mujoco",
                    "logging=detailed_info_evidence_eval_runs",
                    f"++experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self) -> None:
        exp = hydra.utils.instantiate(self.cfg.experiment)
        with exp:
            pass

    def test_logging_detailed_evidence_lm_config_creates_json(self) -> None:
        """Test for the detailed evidence LM logging config.

        This ensures the config can be composed, the experiment can be instantiated,
        and that running an episode produces detailed logging json file.
        """
        exp = hydra.utils.instantiate(self.cfg.experiment)
        with exp:
            exp.model.set_experiment_mode(ExperimentMode.EVAL)
            exp.run_epoch()

        # Detailed logging handler should create a detailed_run_stats.json file
        self.assertTrue(
            (exp.output_dir / "detailed_run_stats.json").exists(),
            "Expected detailed_run_stats.json file to be created",
        )
