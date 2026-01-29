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

import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import hydra

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.profile import (
    ProfileExperimentMixin,
)


class InheritanceProfileExperimentMixinTest(TestCase):
    @staticmethod
    def test_leftmost_subclassing_does_not_error() -> None:
        class GoodSubclass(ProfileExperimentMixin, MontyExperiment):
            pass

    @staticmethod
    def test_non_leftmost_subclassing_raises_error() -> None:
        with pytest.raises(TypeError):

            class BadSubclass(MontyExperiment, ProfileExperimentMixin):
                pass

    @staticmethod
    def test_missing_experiment_base_raises_error() -> None:
        with pytest.raises(TypeError):

            class BadSubclass(ProfileExperimentMixin):
                pass

    @staticmethod
    def test_experiment_subclasses_are_properly_detected() -> None:
        class SubExperiment(MontyExperiment):
            pass

        class Subclass(ProfileExperimentMixin, SubExperiment):
            pass


class ProfiledExperiment(ProfileExperimentMixin, MontyExperiment):
    pass


class ProfileExperimentMixinTest(TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.base_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=profile/base",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def get_profile_files(self) -> set[str]:
        """Helper to get the files in the profile directory in a set.

        Returns:
            Set of filenames in the profile directory.
        """
        path = Path(self.output_dir, "profile")
        filenames = [f.name for f in path.iterdir() if f.is_file()]
        # returning a set so the order doesn't matter
        return set(filenames)

    def spot_check_profile_files(self) -> None:
        """Helper to do some basic asserts on the profile files."""
        path = Path(self.output_dir, "profile")
        for file in path.iterdir():
            if not file.is_file():
                continue
            self.assertGreater(
                file.stat().st_size, 0, "Empty profile file was unexpectedly generated."
            )
            with file.open("r") as f:
                first_line = f.readline().rstrip("\n")
                self.assertEqual(
                    first_line, ",func,ncalls,ccalls,tottime,cumtime,callers"
                )

    def test_run_episode_is_profiled(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.env_interface = exp.train_env_interface
            exp.run_episode()

        self.assertSetEqual(
            self.get_profile_files(),
            {
                "profile-setup_experiment.csv",
                "profile-train_epoch_0_episode_0.csv",
            },
        )
        self.spot_check_profile_files()

    def test_run_train_epoch_is_profiled(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.run_epoch()

        self.assertSetEqual(
            self.get_profile_files(),
            {
                "profile-setup_experiment.csv",
                "profile-train_epoch_0_episode_0.csv",
                "profile-train_epoch_0_episode_1.csv",
                "profile-train_epoch_0_episode_2.csv",
            },
        )
        self.spot_check_profile_files()

    def test_run_eval_epoch_is_profiled(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.EVAL
            exp.model.set_experiment_mode("eval")
            exp.run_epoch()

        self.assertSetEqual(
            self.get_profile_files(),
            {
                "profile-setup_experiment.csv",
                "profile-eval_epoch_0_episode_0.csv",
                "profile-eval_epoch_0_episode_1.csv",
                "profile-eval_epoch_0_episode_2.csv",
            },
        )
        self.spot_check_profile_files()
