# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
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
from pathlib import Path
from pprint import pprint
from typing import Set

import pytest

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    SingleCameraMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    NotYCBEvalObjectList,
    NotYCBTrainObjectList,
)
from tbp.monty.frameworks.environments import embodied_data as ed
from tbp.monty.frameworks.experiments import MontyExperiment, ProfileExperimentMixin
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsSinglePTZ,
    SinglePTZHabitatDatasetArgs,
)


class ProfiledExperiment(ProfileExperimentMixin, MontyExperiment):
    pass


class ProfileExperimentMixinTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()

        base = dict(
            experiment_class=ProfiledExperiment,
            experiment_args=DebugExperimentArgs(),
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG"
            ),
            monty_config=SingleCameraMontyConfig(),
            dataset_class=ed.EnvironmentDataset,
            dataset_args=SinglePTZHabitatDatasetArgs(
                env_init_args=EnvInitArgsSinglePTZ(data_path=None).__dict__
            ),
            train_dataloader_class=ed.EnvironmentDataLoaderPerObject,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=NotYCBTrainObjectList().objects,
            ),
            eval_dataloader_class=ed.EnvironmentDataLoaderPerObject,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=NotYCBEvalObjectList().objects,
            ),
        )

        self.base_config = base

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def get_profile_files(self) -> Set[str]:
        """Helper to get the files in the profile directory in a set.

        Returns:
            set of filenames in the profile directory.
        """
        path = Path(self.output_dir, "profile")
        filenames = [f.name for f in path.iterdir() if f.is_file()]
        # returning a set so the order doesn't matter
        return set(filenames)

    @staticmethod
    def test_correct_subclassing_does_not_error() -> None:
        # this shouldn't raise an exception at evaluation time because
        # the mixin is the leftmost base class listed
        class GoodSubclass(ProfileExperimentMixin, MontyExperiment):
            pass

    @staticmethod
    def test_incorrect_subclassing_raises_error() -> None:
        with pytest.raises(TypeError):
            class BadSubclass(MontyExperiment, ProfileExperimentMixin):
                pass

    @staticmethod
    def test_missing_experiment_base_raises_error() -> None:
        with pytest.raises(TypeError):
            class BadSubclass(ProfileExperimentMixin):
                pass

    def test_can_run_episode(self) -> None:
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        with ProfiledExperiment() as exp:
            exp.setup_experiment(base_config)
            pprint("...training...")
            exp.model.set_experiment_mode("train")
            exp.dataloader = exp.train_dataloader
            exp.run_episode()

        self.assertSetEqual(self.get_profile_files(), {
            "profile-setup_experiment.csv",
            "profile-train_epoch_0_episode_0.csv",
        })

    def test_can_run_train_epoch(self) -> None:
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        with ProfiledExperiment() as exp:
            exp.setup_experiment(base_config)
            exp.model.set_experiment_mode("train")
            exp.run_epoch()

        self.assertSetEqual(self.get_profile_files(), {
            "profile-setup_experiment.csv",
            "profile-train_epoch_0_episode_0.csv",
            "profile-train_epoch_0_episode_1.csv",
            "profile-train_epoch_0_episode_2.csv",
        })

    def test_can_run_eval_epoch(self) -> None:
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        with ProfiledExperiment() as exp:
            exp.setup_experiment(base_config)
            exp.model.set_experiment_mode("eval")
            exp.run_epoch()

        self.assertSetEqual(self.get_profile_files(), {
            "profile-setup_experiment.csv",
            "profile-eval_epoch_0_episode_0.csv",
            "profile-eval_epoch_0_episode_1.csv",
            "profile-eval_epoch_0_episode_2.csv",
        })
