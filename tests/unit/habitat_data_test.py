# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import hydra
import pytest
from omegaconf import OmegaConf

from tbp.monty.frameworks.agents import AgentID
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
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
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

        with hydra.initialize(config_path="../../conf", version_base=None):
            self.policy_cfg_fragment = hydra.compose(
                config_name="experiment/config/monty/motor_system/defaults",
            ).experiment.config.monty.motor_system.motor_system_args.policy_args
            self.policy_cfg_abs_fragment = hydra.compose(
                config_name="test/config/monty/motor_system/absolute",
            ).test.config.monty.motor_system.motor_system_args.policy_args
            self.policy_cfg_surf_fragment = hydra.compose(
                config_name="test/config/monty/motor_system/surface",
            ).test.config.monty.motor_system.motor_system_args.policy_args

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_dist(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_dist = mock_agent_class.return_value
        mock_agent_dist.agent_config = self.camera_dist.get_spec()
        mock_agent_dist.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
        )
        mock_sim_dist = mock_simulator_class.return_value
        mock_sim_dist.agents = [mock_agent_dist]
        mock_sim_dist.get_agent.side_effect = lambda agent_idx: (
            mock_agent_dist if agent_idx == 0 else None
        )
        mock_sim_dist.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)

        # Create distant-agent motor systems / policies
        base_policy_cfg_dist = OmegaConf.to_object(self.policy_cfg_fragment)
        base_policy_cfg_dist["agent_id"] = AGENT_ID

        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_dist)
        )

        # Create habitat env datasets with distant-agent action space
        env_init_args = {"agents": self.camera_dist_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_dist = EnvironmentInterface(
            env, rng=rng, motor_system=motor_system_dist, seed=seed
        )

        # Check if env interface is getting observations from simulator
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_dist, _ = env_interface_dist.step(motor_system_dist())
            camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[MODALITY] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_dist, _ = env_interface_dist.reset()
        initial_camera_obs_dist = initial_obs_dist[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_dist[MODALITY] == EXPECTED_STATES[0]))

        # Check if env interface actions affect simulator observations
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        obs_dist, _ = env_interface_dist.step(motor_system_dist())
        camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_dist[MODALITY] == initial_camera_obs_dist[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_abs(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_abs = mock_agent_class.return_value
        mock_agent_abs.agent_config = self.camera_abs.get_spec()
        mock_agent_abs.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
        )
        mock_sim_abs = mock_simulator_class.return_value
        mock_sim_abs.agents = [mock_agent_abs]
        mock_sim_abs.get_agent.side_effect = lambda agent_idx: (
            mock_agent_abs if agent_idx == 0 else None
        )
        mock_sim_abs.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)

        base_policy_cfg_abs = OmegaConf.to_object(self.policy_cfg_abs_fragment)
        base_policy_cfg_abs["agent_id"] = AGENT_ID

        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_abs)
        )

        # Create habitat env with absolute action space
        env_init_args = {"agents": self.camera_abs_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_abs = EnvironmentInterface(
            env,
            rng=rng,
            motor_system=motor_system_abs,
            seed=seed,
        )

        # Check if env interfaces are getting observations from simulator
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_abs, _ = env_interface_abs.step(motor_system_abs())
            camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[MODALITY] == EXPECTED_STATES[i]))

        # Check env interface reset gets observations from simulator
        initial_obs_abs, _ = env_interface_abs.reset()
        initial_camera_obs_abs = initial_obs_abs[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_abs[MODALITY] == EXPECTED_STATES[0]))

        # Check if env interface actions affect simulator observations
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        obs_abs, _ = env_interface_abs.step(motor_system_abs())
        camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_abs[MODALITY] == initial_camera_obs_abs[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_surf(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_surf = mock_agent_class.return_value
        mock_agent_surf.agent_config = self.camera_surf.get_spec()
        mock_agent_surf.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
        )
        mock_sim_surf = mock_simulator_class.return_value
        mock_sim_surf.agents = [mock_agent_surf]
        mock_sim_surf.get_agent.side_effect = lambda agent_idx: (
            mock_agent_surf if agent_idx == 0 else None
        )
        mock_sim_surf.reset.return_value = self.mock_reset

        seed = 42
        rng = np.random.RandomState(seed)
        # Note we just test random actions (i.e. base policy) with the surface-agent
        # action space
        base_policy_cfg_surf = OmegaConf.to_object(self.policy_cfg_surf_fragment)
        base_policy_cfg_surf["agent_id"] = AGENT_ID

        motor_system_surf = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_surf)
        )

        # Create habitat env interface with distant-agent action space
        env_init_args = {"agents": self.camera_surf_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_surf = EnvironmentInterface(
            env, rng=rng, motor_system=motor_system_surf, seed=seed
        )

        # Check if datasets are getting observations from simulator
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, NUM_STEPS):
            obs_surf, _ = env_interface_surf.step(motor_system_surf())
            camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[MODALITY] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_surf, _ = env_interface_surf.reset()
        initial_camera_obs_surf = initial_obs_surf[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(initial_camera_obs_surf[MODALITY] == EXPECTED_STATES[0]))

        # Check if dataset actions affect simulator observations
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        obs_surf, _ = env_interface_surf.step(motor_system_surf())
        camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_surf[MODALITY] == initial_camera_obs_surf[MODALITY])
        )

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_dist_states(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_dist = mock_agent_class.return_value
        mock_agent_dist.agent_config = self.camera_dist.get_spec()
        mock_agent_dist.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
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

        base_policy_cfg_dist = OmegaConf.to_object(self.policy_cfg_fragment)
        base_policy_cfg_dist["agent_id"] = AGENT_ID
        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_dist)
        )

        env_init_args = {"agents": self.camera_dist_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_dist = EnvironmentInterface(
            env, motor_system=motor_system_dist, rng=rng, seed=seed
        )

        for i, item in enumerate(env_interface_dist):
            camera_obs_dist = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_abs_states(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_abs = mock_agent_class.return_value
        mock_agent_abs.agent_config = self.camera_abs.get_spec()
        mock_agent_abs.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
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

        base_policy_cfg_abs = OmegaConf.to_object(self.policy_cfg_abs_fragment)
        base_policy_cfg_abs["agent_id"] = AGENT_ID
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_abs)
        )
        env_init_args = {"agents": self.camera_abs_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_abs = EnvironmentInterface(
            env, motor_system=motor_system_abs, rng=rng, seed=seed
        )
        for i, item in enumerate(env_interface_abs):
            camera_obs_abs = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_env_interface_surf_states(self, mock_simulator_class, mock_agent_class):
        # Mock habitat_sim classes
        mock_agent_surf = mock_agent_class.return_value
        mock_agent_surf.agent_config = self.camera_surf.get_spec()
        mock_agent_surf.scene_node = mock.Mock(
            rotation=mn.Quaternion.zero_init(), node_sensors={}
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

        # Note we just test random actions (i.e. base policy) with the surface-agent
        # action space
        base_policy_cfg_surf = OmegaConf.to_object(self.policy_cfg_surf_fragment)
        base_policy_cfg_surf["agent_id"] = AGENT_ID
        motor_system_surf = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_cfg_surf)
        )

        env_init_args = {"agents": self.camera_surf_config}
        env = HabitatEnvironment(**env_init_args)
        env_interface_surf = EnvironmentInterface(
            env, motor_system=motor_system_surf, rng=rng, seed=seed
        )
        for i, item in enumerate(env_interface_surf):
            camera_obs_surf = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[MODALITY] == EXPECTED_STATES[i]))
            if i >= NUM_STEPS - 1:
                break


if __name__ == "__main__":
    unittest.main()
