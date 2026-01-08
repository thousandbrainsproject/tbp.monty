# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from unittest import TestCase

import hydra
import pytest

from tbp.monty.frameworks.run import run_name_output_dir

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)


class FistExperimentTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/first_experiment"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()


class TrainingAndInferenceTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/surf_agent_2obj_train"],
            )
            config.experiment.config.logging.output_dir = str(
                run_name_output_dir(config)
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/surf_agent_2obj_eval"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()


class UnsupervisedContinualLearningTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/surf_agent_2obj_unsupervised"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()


class MultipleLearningModulesTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/dist_agent_5lm_2obj_train"],
            )
            config.experiment.config.logging.output_dir = str(
                run_name_output_dir(config)
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/dist_agent_5lm_2obj_eval"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()


class OmniglotTrainingAndInferenceTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/omniglot_training"],
            )
            config.experiment.config.logging.output_dir = str(
                run_name_output_dir(config)
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/omniglot_inference"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

class MontyMeetsWorld2DImageInferenceTest(TestCase):
    def test_tutorial(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/monty_meets_world_2dimage_inference"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()
