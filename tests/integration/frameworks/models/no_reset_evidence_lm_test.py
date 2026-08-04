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

from tbp.monty.hydra import instantiate_experiment
from tests import HYDRA_ROOT

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import shutil
import tempfile
from typing import Any

import hydra
import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests.unit.resources.unit_test_utils import BaseGraphTest


class NoResetEvidenceLMTest(BaseGraphTest):
    def setUp(self):
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.pretraining_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/no_reset_evidence_lm/pretraining",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.unsupervised_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/no_reset_evidence_lm/unsupervised",
                    f"experiment.config.logging.output_dir={self.output_dir}",
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
        train_exp = instantiate_experiment(self.pretraining_cfg.experiment)
        with train_exp:
            train_exp.run()

        eval_exp = instantiate_experiment(self.unsupervised_cfg.experiment)
        with eval_exp:
            # load the eval experiment with the pretrained models
            pretrained_models = train_exp.model.learning_modules[0].state_dict()
            eval_exp.model.learning_modules[0].load_state_dict(pretrained_models)

            # This test assumes that the Monty instance, and the learning modules
            # in particular, remain stable throughout the test, which is the case
            # when using the "reset" strategy. However, with the "recreation" strategy
            # we get a new instance each time `pre_episode()` is called, so we use
            # the internal `_snapshot_monty()` method to prepare a Memento for use
            # in re-creating Monty.
            # TODO: redesign experiments to use Memento snapshots to set Monty to
            # the state needed for the test and avoid using the "reset" mechanism.
            eval_exp._snapshot_monty()

            eval_exp.experiment_mode = ExperimentMode.EVAL
            eval_exp.model.set_experiment_mode(eval_exp.experiment_mode)
            eval_exp.pre_epoch()

            # first episode
            eval_exp.pre_episode()
            lm = eval_exp.model.learning_modules[0]
            self.assertEqual(
                len(lm._hypotheses),
                0,
                "evidence dict should be empty before the first episode",
            )
            episode_1_steps = eval_exp.run_episode_steps()
            eval_exp.post_episode(episode_1_steps)
            post_episode1_evidence = copy.deepcopy(
                {graph_id: hyp.evidence for graph_id, hyp in lm._hypotheses.items()}
            )
            self.assertGreater(
                len(post_episode1_evidence),
                0,
                "evidence dict should now contain evidence values of the first episode",
            )

            # second episode
            eval_exp.pre_episode()
            lm = eval_exp.model.learning_modules[0]
            self.assert_dicts_equal(
                post_episode1_evidence,
                {graph_id: hyp.evidence for graph_id, hyp in lm._hypotheses.items()},
                "evidence dict should not change between episodes",
            )
            episode_2_steps = eval_exp.run_episode_steps()
            eval_exp.post_episode(episode_2_steps)
            self.assertGreater(
                len(lm._hypotheses),
                0,
                "evidence dict should contain evidence values",
            )
            eval_exp.post_epoch()

    def tearDown(self):
        shutil.rmtree(self.output_dir)
