# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import copy
import shutil
import tempfile
import unittest
from pprint import pprint

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.evidence_unsupervised_inference_matching import (
    MontyForUnsupervisedEvidenceGraphMatching,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)
from tests.unit.resources.unit_test_utils import BaseGraphTestCases


class UnsInfEvidenceLMTest(BaseGraphTestCases.BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        default_tolerances = {
            "hsv": np.array([0.1, 1, 1]),
            "principal_curvatures_log": np.ones(2),
        }

        default_lm_args = dict(
            max_match_distance=0.001,
            tolerances={"patch": default_tolerances},
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
        )

        default_evidence_lm_config = dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=default_lm_args,
            )
        )

        self.output_dir = tempfile.mkdtemp()

        self.evidence_config = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                max_train_steps=30, max_eval_steps=30, max_total_steps=60
            ),
            # NOTE: could make unit tests faster by setting monty_log_level="BASIC" for
            # some of them.
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG"
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_class=MontyForEvidenceGraphMatching,
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=default_evidence_lm_config,
            ),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.unsupervised_evidence_config = copy.deepcopy(self.evidence_config)
        self.unsupervised_evidence_config[
            "monty_config"
        ].monty_class = MontyForUnsupervisedEvidenceGraphMatching

    def test_can_reset_buffer_evidence_lm(self):
        """Checks that the default EvidenceGraphLM resets the buffer between episodes.

        This test uses the `self.evidence_config` which defines
        `MontyForEvidenceGraphMatching` as the Monty Class. The expected behavior
        is that the buffer is reset between episodes. This forces the hypothesis
        space to be reinintialized when `EvidenceGraphLM._update_evidence` is called.
        """

        def run_episode():
            for step, observation in enumerate(self.exp.dataloader):
                self.exp.pre_step(step, observation)
                self.exp.model.step(observation)
                self.exp.post_step(step, observation)
                if self.exp.model.is_done or step >= self.exp.max_steps:
                    break

            return step

        pprint("...parsing experiment...")
        config = copy.deepcopy(self.evidence_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...evaluating...")

            self.exp.model.set_experiment_mode("eval")

            self.exp.pre_epoch()

            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                0,
                "Buffer should be empty at the beginning of the episode",
            )
            self.exp.pre_episode()
            episode_1_steps = run_episode()
            self.exp.post_episode(episode_1_steps)

            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                episode_1_steps + 1,
                "Buffer should contain the number of evaluation steps for episode 1.",
            )
            self.exp.pre_episode()
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                0,
                "Buffer should reset at the beginning of a new episode",
            )
            episode_2_steps = run_episode()
            self.exp.post_episode(episode_2_steps)
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                episode_2_steps + 1,
                "Buffer should contain the number of evaluation steps for episode 2.",
            )

            self.assertEqual(len(self.exp.model.learning_modules[0].buffer), 4)
            self.exp.post_epoch()
            self.exp.close()

    def test_no_reset_buffer_unsupervised_evidence_lm(self):
        """Checks that unsupervised LM does not reset the buffer between episodes.

        This test uses the `self.unsupervised_evidence_config` which defines
        `MontyForUnsupervisedEvidenceGraphMatching` as the Monty Class. The expected
        behavior is that the buffer is not reset between episodes.

        Note: We cannot currently test the evidence directly because the evidence is
        not explicitly reset at the beginning of the episode. Instead the buffer is
        reset, which causes displacements to become None (len(buffer) <=1), and
        hypotheses to be overwritten.
        """

        def run_episode():
            for step, observation in enumerate(self.exp.dataloader):
                self.exp.pre_step(step, observation)
                self.exp.model.step(observation)
                self.exp.post_step(step, observation)
                if self.exp.model.is_done or step >= self.exp.max_steps:
                    break

            return step

        pprint("...parsing experiment...")
        config = copy.deepcopy(self.unsupervised_evidence_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...evaluating...")

            self.exp.model.set_experiment_mode("eval")

            self.exp.pre_epoch()

            # first episode
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                0,
                "Buffer should be empty at the beginning of the episode",
            )
            self.exp.pre_episode()
            episode_1_steps = run_episode()
            self.exp.post_episode(episode_1_steps)
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                episode_1_steps + 1,
                "Buffer should contain the number of evaluation steps for episode 1.",
            )

            # second episode
            self.exp.pre_episode()
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                episode_1_steps + 1,
                "Buffer should not reset at the beginning of a new episode",
            )
            episode_2_steps = run_episode()
            self.exp.post_episode(episode_2_steps)
            self.assertEqual(
                len(self.exp.model.learning_modules[0].buffer),
                episode_1_steps + episode_2_steps + 2,
                "Buffer should contain the number of evaluation steps for 2 episodes.",
            )
            self.exp.post_epoch()
            self.exp.close()

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)


if __name__ == "__main__":
    unittest.main()
