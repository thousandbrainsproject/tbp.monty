# Copyright 2025-2026 Thousand Brains Project
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
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.sensors import SensorID

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import unittest
from unittest import mock

import magnum as mn
import numpy as np

from tbp.monty.frameworks.environments.embodied_data import EnvironmentInterface
from tbp.monty.simulators.habitat import SingleSensorAgent
from tbp.monty.simulators.habitat.environment import AgentConfig, HabitatEnvironment

NUM_STEPS = 10
DEFAULT_ACTUATION_AMOUNT = 0.25
AGENT_ID = AgentID("camera")
SENSOR_ID = SensorID("sensor_id_0")
MODALITY = "depth"
EXPECTED_STATES = np.random.rand(NUM_STEPS, 64, 64, 1)


class HabitatDataTest(unittest.TestCase):
    def setUp(self):
        self.camera_dist_config = AgentConfig(
            SingleSensorAgent, dict(agent_id=AGENT_ID, sensor_id=SENSOR_ID)
        )
        self.camera_dist = SingleSensorAgent(agent_id=AGENT_ID, sensor_id=SENSOR_ID)
        self.camera_abs_config = AgentConfig(
            SingleSensorAgent,
            dict(
                agent_id=AGENT_ID,
                sensor_id=SENSOR_ID,
                action_space_type="absolute_only",
            ),
        )
        self.camera_abs = SingleSensorAgent(
            agent_id=AGENT_ID, sensor_id=SENSOR_ID, action_space_type="absolute_only"
        )
        self.camera_surf_config = AgentConfig(
            SingleSensorAgent,
            dict(
                agent_id=AGENT_ID,
                sensor_id=SENSOR_ID,
                action_space_type="surface_agent",
            ),
        )
        self.camera_surf = SingleSensorAgent(
            agent_id=AGENT_ID, sensor_id=SENSOR_ID, action_space_type="surface_agent"
        )
        self.mock_reset = {0: {f"{SENSOR_ID}.depth": EXPECTED_STATES[0]}}
        self.mock_observations = [
            {0: {f"{SENSOR_ID}.depth": s}} for s in EXPECTED_STATES[1:]
        ]

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_dist(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_dist = mock_agent_class.return_value
        mock_agent_dist.agent_config = self.camera_dist.get_spec()
        mock_agent_dist.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_dist = mock_simulator_class.return_value
        mock_sim_dist.agents = [mock_agent_dist]
        mock_sim_dist.get_agent.side_effect = lambda agent_idx: (
            mock_agent_dist if agent_idx == 0 else None
        )
        mock_sim_dist.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)

        # Create habitat env datasets with distant-agent action space
        env_init_args = {"agents": self.camera_dist_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_dist = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Check if env interface is getting observations from simulator
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_dist, _ = env_interface_dist.step([])
            camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[MODALITY] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_dist, _ = env_interface_dist.reset(rng)
        initial_camera_obs_dist = initial_obs_dist[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_dist[MODALITY] == EXPECTED_STATES[0]))

        # Check if env interface actions affect simulator observations
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        obs_dist, _ = env_interface_dist.step([])
        camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_dist[MODALITY] == initial_camera_obs_dist[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_abs(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_abs = mock_agent_class.return_value
        mock_agent_abs.agent_config = self.camera_abs.get_spec()
        mock_agent_abs.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_abs = mock_simulator_class.return_value
        mock_sim_abs.agents = [mock_agent_abs]
        mock_sim_abs.get_agent.side_effect = lambda agent_idx: (
            mock_agent_abs if agent_idx == 0 else None
        )
        mock_sim_abs.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)

        # Create habitat env with absolute action space
        env_init_args = {"agents": self.camera_abs_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_abs = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Check if env interfaces are getting observations from simulator
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_abs, _ = env_interface_abs.step([])
            camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[MODALITY] == EXPECTED_STATES[i]))

        # Check env interface reset gets observations from simulator
        initial_obs_abs, _ = env_interface_abs.reset(rng)
        initial_camera_obs_abs = initial_obs_abs[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_abs[MODALITY] == EXPECTED_STATES[0]))

        # Check if env interface actions affect simulator observations
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        obs_abs, _ = env_interface_abs.step([])
        camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_abs[MODALITY] == initial_camera_obs_abs[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_surf(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_surf = mock_agent_class.return_value
        mock_agent_surf.agent_config = self.camera_surf.get_spec()
        mock_agent_surf.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_surf = mock_simulator_class.return_value
        mock_sim_surf.agents = [mock_agent_surf]
        mock_sim_surf.get_agent.side_effect = lambda agent_idx: (
            mock_agent_surf if agent_idx == 0 else None
        )
        mock_sim_surf.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)

        # Create habitat env interface with distant-agent action space
        env_init_args = {"agents": self.camera_surf_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_surf = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Check if datasets are getting observations from simulator
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_surf, _ = env_interface_surf.step([])
            camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[MODALITY] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_surf, _ = env_interface_surf.reset(rng)
        initial_camera_obs_surf = initial_obs_surf[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_surf[MODALITY] == EXPECTED_STATES[0]))

        # Check if dataset actions affect simulator observations
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        obs_surf, _ = env_interface_surf.step([])
        camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_surf[MODALITY] == initial_camera_obs_surf[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_dist_states(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_dist = mock_agent_class.return_value
        mock_agent_dist.agent_config = self.camera_dist.get_spec()
        mock_agent_dist.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_dist = mock_simulator_class.return_value
        mock_sim_dist.agents = [mock_agent_dist]
        mock_sim_dist.get_agent.side_effect = lambda agent_idx: (
            mock_agent_dist if agent_idx == 0 else None
        )
        mock_sim_dist.reset.return_value = self.mock_reset
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations

        seed = 42
        rng = np.random.RandomState(seed)

        env_init_args = {"agents": self.camera_dist_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_dist = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Start at 1 because the initial call to reset consumes the zeroth state.
        i = 1
        while True:
            obs, _ = env_interface_dist.step([])
            camera_obs_dist = obs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break

            i += 1

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_abs_states(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_abs = mock_agent_class.return_value
        mock_agent_abs.agent_config = self.camera_abs.get_spec()
        mock_agent_abs.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_abs = mock_simulator_class.return_value
        mock_sim_abs.agents = [mock_agent_abs]
        mock_sim_abs.get_agent.side_effect = lambda agent_idx: (
            mock_agent_abs if agent_idx == 0 else None
        )
        mock_sim_abs.reset.return_value = self.mock_reset
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations

        seed = 42
        rng = np.random.RandomState(seed)

        env_init_args = {"agents": self.camera_abs_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_abs = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )
        # Start at 1 because the initial call to reset consumes the zeroth state.
        i = 1
        while True:
            obs, _ = env_interface_abs.step([])
            camera_obs_abs = obs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break

            i += 1

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_surf_states(
        self,
        mock_simulator_class: mock.MagicMock,
        mock_agent_class: mock.MagicMock,
    ):
        # Mock habitat_sim classes
        mock_agent_surf = mock_agent_class.return_value
        mock_agent_surf.agent_config = self.camera_surf.get_spec()
        mock_agent_surf.scene_node = mock.Mock(
            translation=mn.Vector3.zero_init(),
            rotation=mn.Quaternion.zero_init(),
            node_sensors={},
        )
        mock_sim_surf = mock_simulator_class.return_value
        mock_sim_surf.agents = [mock_agent_surf]
        mock_sim_surf.get_agent.side_effect = lambda agent_idx: (
            mock_agent_surf if agent_idx == 0 else None
        )
        mock_sim_surf.reset.return_value = self.mock_reset
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations

        seed = 42
        rng = np.random.RandomState(seed)

        env_init_args = {"agents": self.camera_surf_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_surf = EnvironmentInterface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )
        # Start at 1 because the initial call to reset consumes the zeroth state.
        i = 1
        while True:
            obs, _ = env_interface_surf.step([])
            camera_obs_surf = obs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break

            i += 1
