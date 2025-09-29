# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import shutil
import tempfile
import unittest
from typing import Any, Dict

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.no_reset_evidence_matching import (
    MontyForNoResetEvidenceGraphMatching,
    NoResetEvidenceGraphLM,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)
from tests.unit.resources.unit_test_utils import BaseGraphTestCases


class NoResetEvidenceLMTest(BaseGraphTestCases.BaseGraphTest):
    def setUp(self):
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

        default_unsupervised_evidence_lm_config = dict(
            learning_module_0=dict(
                learning_module_class=NoResetEvidenceGraphLM,
                learning_module_args=default_lm_args,
            )
        )

        self.output_dir = tempfile.mkdtemp()

        self.pretraining_configs = dict(
            experiment_class=MontySupervisedObjectPretrainingExperiment,
            experiment_args=ExperimentArgs(
                do_eval=False,
                n_train_epochs=3,
            ),
            logging_config=PretrainLoggingConfig(
                output_dir=self.output_dir,
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_class=MontyForEvidenceGraphMatching,
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=default_evidence_lm_config,
            ),
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.unsupervised_evidence_config = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                max_train_steps=30, max_eval_steps=30, max_total_steps=60
            ),
            # NOTE: could make unit tests faster by setting monty_log_level="BASIC" for
            # some of them.
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG", monty_handlers=[]
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_class=MontyForNoResetEvidenceGraphMatching,
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=default_unsupervised_evidence_lm_config,
            ),
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

    def assert_dicts_equal(
        self, d1: Dict[Any, np.ndarray], d2: Dict[Any, np.ndarray], msg: str
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
        train_config = copy.deepcopy(self.pretraining_configs)
        with MontySupervisedObjectPretrainingExperiment(train_config) as train_exp:
            train_exp.train()

        eval_config = copy.deepcopy(self.unsupervised_evidence_config)
        with MontyObjectRecognitionExperiment(eval_config) as eval_exp:
            # load the eval experiment with the pretrained models
            pretrained_models = train_exp.model.learning_modules[0].state_dict()
            eval_exp.model.learning_modules[0].load_state_dict(pretrained_models)

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
