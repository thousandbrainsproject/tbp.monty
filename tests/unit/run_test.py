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

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hydra
import magnum as mn
import numpy as np
from omegaconf import OmegaConf

from tbp.monty.frameworks.run import main
from tbp.monty.simulators.habitat import SingleSensorAgent

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
        self.output_dir = tempfile.mkdtemp()
        agent_patch = mock.patch("habitat_sim.Agent", autospec=True)
        sim_patch = mock.patch("habitat_sim.Simulator", autospec=True)
        self.addCleanup(agent_patch.stop)
        self.addCleanup(sim_patch.stop)

        # Mock habitat_sim classes
        mock_agent_class = agent_patch.start()
        camera = SingleSensorAgent(
            agent_id=AgentID("agent_id_0"), sensor_id=SensorID("sensor_id_0")
        )
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

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/run",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_main(self):
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        main(self.cfg)

        output_dir = Path(self.cfg.experiment.config.logging.output_dir)

        with (output_dir / "fake_log.json").open("r") as f:
            exp_log = json.load(f)

        self.assertListEqual(exp_log, EXPECTED_LOG)


if __name__ == "__main__":
    unittest.main()
