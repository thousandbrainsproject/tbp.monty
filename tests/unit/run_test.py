# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
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

import os
import pathlib
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import magnum as mn
import numpy as np

from tbp.monty.frameworks.config_utils.config_args import LoggingConfig
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
)
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
)
from tbp.monty.frameworks.experiments import MontyExperiment
from tbp.monty.frameworks.run import main, run
from tbp.monty.simulators.habitat import SingleSensorAgent
from tbp.monty.simulators.habitat.configs import (
    SinglePTZHabitatDatasetArgs,
)
from tests.unit.frameworks.config_utils.fakes.config_args import (
    FakeSingleCameraMontyConfig,
)

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
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()
        agent_patch = mock.patch("habitat_sim.Agent", autospec=True)
        sim_patch = mock.patch("habitat_sim.Simulator", autospec=True)
        self.addCleanup(agent_patch.stop)
        self.addCleanup(sim_patch.stop)

        # Mock habitat_sim classes
        mock_agent_class = agent_patch.start()
        camera = SingleSensorAgent(agent_id="agent_id_0", sensor_id="sensor_id_0")
        self.mock_agent = mock_agent_class.return_value
        self.mock_agent.agent_config = camera.get_spec()
        self.mock_agent.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
        )
        mock_sim_class = sim_patch.start()
        self.mock_sim = mock_sim_class.return_value
        self.mock_sim.agents = [self.mock_agent]
        self.mock_sim.get_agent.side_effect = lambda agent_idx: (
            self.mock_agent if agent_idx == 0 else None
        )
        self.mock_sim.reset.return_value = {
            0: {"agent_id_0.depth": np.random.rand(64, 64, 1)}
        }
        self.mock_sim.get_sensor_observations.side_effect = [
            {0: {"agent_id_0.depth": obs}} for obs in FAKE_OBS
        ]

        self.CONFIGS = {
            "test_1": {
                "experiment_class": MontyExperiment,
                "experiment_args": ExperimentArgs(
                    max_train_steps=MAX_TRAIN_STEPS,
                    max_eval_steps=MAX_EVAL_STEPS,
                    n_train_epochs=TRAIN_EPOCHS,
                    n_eval_epochs=EVAL_EPOCHS,
                ),
                "logging_config": LoggingConfig(
                    output_dir=self.output_dir,
                    monty_log_level="TEST",
                    monty_handlers=[],
                ),
                "monty_config": FakeSingleCameraMontyConfig(),
                "dataset_args": SinglePTZHabitatDatasetArgs(),
                "train_dataloader_class": EnvironmentDataLoader,
                "train_dataloader_args": {},
                "eval_dataloader_class": EnvironmentDataLoader,
                "eval_dataloader_args": {},
            },
            "test_2": {
                "experiment_class": MontyExperiment,
                "experiment_args": ExperimentArgs(
                    max_train_steps=MAX_TRAIN_STEPS,
                    max_eval_steps=MAX_EVAL_STEPS,
                    n_train_epochs=TRAIN_EPOCHS,
                    n_eval_epochs=EVAL_EPOCHS,
                ),
                "logging_config": LoggingConfig(
                    output_dir=self.output_dir,
                    monty_log_level="TEST",
                    monty_handlers=[],
                ),
                "monty_config": FakeSingleCameraMontyConfig(),
                "dataset_args": SinglePTZHabitatDatasetArgs(),
                "train_dataloader_class": EnvironmentDataLoader,
                "train_dataloader_args": {},
                "eval_dataloader_class": EnvironmentDataLoader,
                "eval_dataloader_args": {},
            },
        }

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_main_without_experiment(self):
        sys.argv = ["monty", "-e", "test_1"]
        main(all_configs=self.CONFIGS)

        output_dir = os.path.join(
            self.CONFIGS["test_1"]["logging_config"].output_dir, "test_1"
        )
        with open(os.path.join(output_dir, "fake_log.pkl"), "rb") as f:
            exp_log = pickle.load(f)

        self.assertListEqual(exp_log, EXPECTED_LOG)

    def test_main_with_single_experiment(self):
        main(all_configs=self.CONFIGS, experiments=["test_1"])

        output_dir = os.path.join(
            self.CONFIGS["test_1"]["logging_config"].output_dir, "test_1"
        )
        with open(os.path.join(output_dir, "fake_log.pkl"), "rb") as f:
            exp_log = pickle.load(f)

        self.assertListEqual(exp_log, EXPECTED_LOG)

    def test_main_with_multiple_experiment(self):
        main(all_configs=self.CONFIGS, experiments=["test_1", "test_2"])

        output_dir_1 = os.path.join(
            self.CONFIGS["test_1"]["logging_config"].output_dir, "test_1"
        )
        with open(os.path.join(output_dir_1, "fake_log.pkl"), "rb") as f:
            exp_log_1 = pickle.load(f)

        self.assertListEqual(exp_log_1, EXPECTED_LOG)

        output_dir_2 = os.path.join(
            self.CONFIGS["test_2"]["logging_config"].output_dir, "test_2"
        )
        with open(os.path.join(output_dir_2, "fake_log.pkl"), "rb") as f:
            exp_log_2 = pickle.load(f)

        self.assertListEqual(exp_log_2, EXPECTED_LOG)

    def test_run(self):
        run(config=self.CONFIGS["test_1"])

        output_dir = os.path.join(self.CONFIGS["test_1"]["logging_config"].output_dir)
        with open(os.path.join(output_dir, "fake_log.pkl"), "rb") as f:
            exp_log = pickle.load(f)

        self.assertListEqual(exp_log, EXPECTED_LOG)

    def test_importing_existing_configs(self):
        """Test that any changes do not cause errors in existing configs."""
        current_file_path = pathlib.Path(__file__).parent.resolve()
        base_repo_idx = current_file_path.parts.index("tests")
        while current_file_path.parts[base_repo_idx + 1 :].count("tests") > 0:
            # We want the _last_ "tests" index
            base_repo_idx += 1
        base_dir = pathlib.Path(*current_file_path.parts[:base_repo_idx])
        exp_dir = os.path.join(str(base_dir), "benchmarks")
        sys.path.append(exp_dir)
        import configs  # noqa: F401


if __name__ == "__main__":
    unittest.main()
