# Copyright 2025-2026 Thousand Brains Project
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

import copy
import shutil
import tempfile
import unittest
from typing import Any

import hydra
import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests.unit.resources.unit_test_utils import BaseGraphTest


class NoResetEvidenceLMTest(BaseGraphTest):
    def setUp(self):
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.pretraining_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=no_reset_evidence_lm/pretraining",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.unsupervised_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=no_reset_evidence_lm/unsupervised",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )

    def assert_dicts_equal(
        self, d1: dict[Any, np.ndarray], d2: dict[Any, np.ndarray], msg: str
    ) -> None:
        """Asserts that two dictionaries containing NumPy arrays are equal.

        This method checks that the dictionaries have the same keys and that
        the corresponding NumPy arrays are element-wise equal.

        Args:
            d1: The first dictionary to compare.
            d2: The second dictionary to compare.
            msg: The message to display if the assertion fails.
        """
        self.assertEqual(d1.keys(), d2.keys(), msg)
        for key, d1_val in d1.items():
            self.assertTrue(np.array_equal(d1_val, d2[key]), msg)

    def test_no_reset_evidence_lm(self):
        """Checks that unsupervised LM does not reset the evidence between episodes.

        This test uses the `self.unsupervised_evidence_config` which defines
        `MontyForNoResetEvidenceGraphMatching` and `NoResetEvidenceGraphLM`
        as the Monty Class and Monty LM Class, respectively. The expected behavior is
        that the evidence values are not reset or changed between episodes.

        Note: We use the default `MontyForEvidenceGraphMatching` and `EvidenceGraphLM`
        to train a Monty Experiment, then transfer the pretrained graphs to an
        unsupervised Inference Experiment. Disabling the reset logic does not support
        training at the moment.
        """
        train_exp = hydra.utils.instantiate(self.pretraining_cfg.test)
        with train_exp:
            train_exp.run()

        eval_exp = hydra.utils.instantiate(self.unsupervised_cfg.test)
        with eval_exp:
            # load the eval experiment with the pretrained models
            pretrained_models = train_exp.model.learning_modules[0].state_dict()
            eval_exp.model.learning_modules[0].load_state_dict(pretrained_models)

            eval_exp.experiment_mode = ExperimentMode.EVAL
            eval_exp.model.set_experiment_mode("eval")
            eval_exp.pre_epoch()

            # first episode
            self.assertEqual(
                len(eval_exp.model.learning_modules[0].evidence),
                0,
                "evidence dict should be empty before the first episode",
            )
            eval_exp.pre_episode()
            episode_1_steps = eval_exp.run_episode_steps()
            eval_exp.post_episode(episode_1_steps)
            post_episode1_evidence = copy.deepcopy(
                eval_exp.model.learning_modules[0].evidence
            )
            self.assertGreater(
                len(post_episode1_evidence),
                0,
                "evidence dict should now contain evidence values of the first episode",
            )

            # second episode
            eval_exp.pre_episode()
            self.assert_dicts_equal(
                post_episode1_evidence,
                eval_exp.model.learning_modules[0].evidence,
                "evidence dict should not change between episodes",
            )
            episode_2_steps = eval_exp.run_episode_steps()
            eval_exp.post_episode(episode_2_steps)
            self.assertGreater(
                len(eval_exp.model.learning_modules[0].evidence),
                0,
                "evidence dict should contain evidence values",
            )
            eval_exp.post_epoch()

    def tearDown(self):
        shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
