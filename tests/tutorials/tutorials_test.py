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

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent.parent / "conf" / "experiment"
TUTORIALS_DIR = EXPERIMENT_DIR / "tutorial"


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
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            for path in Path(config.experiment.config.logging.output_dir).iterdir():
                print(path.absolute())

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
