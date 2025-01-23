# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
import unittest.mock as mock

import magnum as mn
import numpy as np

from tbp.monty.frameworks.actions.action_samplers import (
    UniformlyDistributedSampler,
)
from tbp.monty.frameworks.config_utils.config_args import make_base_policy_config
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
    EnvironmentDataset,
)
from tbp.monty.frameworks.environments.embodied_environment import ActionSpace
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.simulators.habitat import SingleSensorAgent
from tbp.monty.simulators.habitat.environment import AgentConfig, HabitatEnvironment

DATASET_LEN = 10
DEFAULT_ACTUATION_AMOUNT = 0.25
AGENT_ID = "camera"
SENSOR_ID = "sensor_id_0"
SENSORS = ["depth"]
EXPECTED_STATES = np.random.rand(DATASET_LEN, 64, 64, 1)
EXPECTED_ACTIONS_DIST = (
    f"{AGENT_ID}.look_down",
    f"{AGENT_ID}.look_up",
    f"{AGENT_ID}.move_forward",
    f"{AGENT_ID}.turn_left",
    f"{AGENT_ID}.turn_right",
    f"{AGENT_ID}.set_agent_pose",
    f"{AGENT_ID}.set_sensor_rotation",
)
EXPECTED_ACTIONS_ABS = (
    f"{AGENT_ID}.set_yaw",
    f"{AGENT_ID}.set_agent_pitch",
    f"{AGENT_ID}.set_sensor_pitch",
    f"{AGENT_ID}.set_agent_pose",
    f"{AGENT_ID}.set_sensor_rotation",
    f"{AGENT_ID}.set_sensor_pose",
)
EXPECTED_ACTIONS_SURF = (
    f"{AGENT_ID}.move_forward",
    f"{AGENT_ID}.move_tangentially",
    f"{AGENT_ID}.orient_horizontal",
    f"{AGENT_ID}.orient_vertical",
    f"{AGENT_ID}.set_agent_pose",
    f"{AGENT_ID}.set_sensor_rotation",
)


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
    def test_dataset_dist(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        # Create habitat env datasets with distant-agent action space
        dataset_dist = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_dist_config]),
            rng=rng,
        )

        # Check distant-agent action space
        action_space_dist = dataset_dist.action_space
        action_space_dist.rng = rng
        self.assertIsInstance(action_space_dist, ActionSpace)
        self.assertCountEqual(action_space_dist, EXPECTED_ACTIONS_DIST)
        self.assertIn(action_space_dist.sample(), EXPECTED_ACTIONS_DIST)

        # Create distant-agent motor systems / policies
        motor_system_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = BasePolicy(rng=rng, **motor_system_config_dist.__dict__)

        # Check if datasets are getting observations from simulator
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, DATASET_LEN):
            obs_dist, _ = dataset_dist[motor_system_dist()]
            camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[SENSORS[0]] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_dist, _ = dataset_dist.reset()
        initial_camera_obs_dist = initial_obs_dist[AGENT_ID][SENSOR_ID]
        self.assertTrue(
            np.all(initial_camera_obs_dist[SENSORS[0]] == EXPECTED_STATES[0])
        )

        # Check if dataset actions affect simulator observations
        mock_sim_dist.get_sensor_observations.side_effect = self.mock_observations
        obs_dist, _ = dataset_dist[motor_system_dist()]
        camera_obs_dist = obs_dist[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_dist[SENSORS[0]] == initial_camera_obs_dist[SENSORS[0]])
        )

        dataset_dist.close()
        with self.assertRaises(Exception):  # noqa: B017
            _ = dataset_dist[action_space_dist.sample()]

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_dataset_abs(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        # Create habitat env datasets with absolute action space
        dataset_abs = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_abs_config]),
            rng=rng,
        )

        # Check absolute action space
        action_space_abs = dataset_abs.action_space
        action_space_abs.rng = rng
        self.assertIsInstance(action_space_abs, ActionSpace)
        self.assertCountEqual(action_space_abs, EXPECTED_ACTIONS_ABS)
        self.assertIn(action_space_abs.sample(), EXPECTED_ACTIONS_ABS)

        motor_system_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = BasePolicy(rng=rng, **motor_system_config_abs.__dict__)

        # Check if datasets are getting observations from simulator
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, DATASET_LEN):
            obs_abs, _ = dataset_abs[motor_system_abs()]
            camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[SENSORS[0]] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_abs, _ = dataset_abs.reset()
        initial_camera_obs_abs = initial_obs_abs[AGENT_ID][SENSOR_ID]
        self.assertTrue(
            np.all(initial_camera_obs_abs[SENSORS[0]] == EXPECTED_STATES[0])
        )

        # Check if dataset actions affect simulator observations
        mock_sim_abs.get_sensor_observations.side_effect = self.mock_observations
        obs_abs, _ = dataset_abs[motor_system_abs()]
        camera_obs_abs = obs_abs[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_abs[SENSORS[0]] == initial_camera_obs_abs[SENSORS[0]])
        )

        dataset_abs.close()
        with self.assertRaises(Exception):  # noqa: B017
            _ = dataset_abs[action_space_abs.sample()]

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_dataset_surf(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        # Create habitat env datasets with distant-agent action space
        dataset_surf = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_surf_config]),
            rng=rng,
        )

        # Check surface-agent action space
        action_space_surf = dataset_surf.action_space
        action_space_surf.rng = rng
        self.assertIsInstance(action_space_surf, ActionSpace)
        self.assertCountEqual(action_space_surf, EXPECTED_ACTIONS_SURF)
        self.assertIn(action_space_surf.sample(), EXPECTED_ACTIONS_SURF)

        # Note we just test random actions (i.e. base policy) with the surface-agent
        # action space
        motor_system_config_surf = make_base_policy_config(
            action_space_type="surface_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_surf = BasePolicy(rng=rng, **motor_system_config_surf.__dict__)

        # Check if datasets are getting observations from simulator
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        for i in range(1, DATASET_LEN):
            obs_surf, _ = dataset_surf[motor_system_surf()]
            camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[SENSORS[0]] == EXPECTED_STATES[i]))

        # Check dataset reset gets observations from simulator
        initial_obs_surf, _ = dataset_surf.reset()
        initial_camera_obs_surf = initial_obs_surf[AGENT_ID][SENSOR_ID]
        self.assertTrue(
            np.all(initial_camera_obs_surf[SENSORS[0]] == EXPECTED_STATES[0])
        )

        # Check if dataset actions affect simulator observations
        mock_sim_surf.get_sensor_observations.side_effect = self.mock_observations
        obs_surf, _ = dataset_surf[motor_system_surf()]
        camera_obs_surf = obs_surf[AGENT_ID][SENSOR_ID]
        self.assertFalse(
            np.all(camera_obs_surf[SENSORS[0]] == initial_camera_obs_surf[SENSORS[0]])
        )

        dataset_surf.close()
        with self.assertRaises(Exception):  # noqa: B017
            _ = dataset_surf[action_space_surf.sample()]

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_dataloader_dist(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        dataset_dist = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_dist_config]),
            rng=rng,
        )

        motor_system_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = BasePolicy(rng=rng, **motor_system_config_dist.__dict__)

        dataloader_dist = EnvironmentDataLoader(dataset_dist, motor_system_dist, rng)
        initial_obs_dist = next(dataloader_dist)
        camera_obs_dist = initial_obs_dist[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(camera_obs_dist[SENSORS[0]] == EXPECTED_STATES[0]))

        for i, item in enumerate(dataloader_dist):
            camera_obs_dist = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_dist[SENSORS[0]] == EXPECTED_STATES[i]))
            if i >= DATASET_LEN - 1:
                break

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_dataloader_abs(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        dataset_abs = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_abs_config]),
            rng=rng,
        )

        motor_system_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = BasePolicy(rng=rng, **motor_system_config_abs.__dict__)

        dataloader_abs = EnvironmentDataLoader(dataset_abs, motor_system_abs, rng)
        initial_obs_abs = next(dataloader_abs)
        camera_obs_abs = initial_obs_abs[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(camera_obs_abs[SENSORS[0]] == EXPECTED_STATES[0]))

        for i, item in enumerate(dataloader_abs):
            camera_obs_abs = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_abs[SENSORS[0]] == EXPECTED_STATES[i]))
            if i >= DATASET_LEN - 1:
                break

    @mock.patch("habitat_sim.Agent", autospec=True)
    @mock.patch("habitat_sim.Simulator", autospec=True)
    def test_dataloader_surf(self, mock_simulator_class, mock_agent_class):
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

        rng = np.random.RandomState(42)

        dataset_surf = EnvironmentDataset(
            env_init_func=HabitatEnvironment,
            env_init_args=dict(agents=[self.camera_surf_config]),
            rng=rng,
        )

        # Note we just test random actions (i.e. base policy) with the surface-agent
        # action space
        motor_system_config_surf = make_base_policy_config(
            action_space_type="surface_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_surf = BasePolicy(rng=rng, **motor_system_config_surf.__dict__)

        dataloader_surf = EnvironmentDataLoader(dataset_surf, motor_system_surf, rng)
        initial_obs_surf = next(dataloader_surf)
        camera_obs_surf = initial_obs_surf[AGENT_ID][SENSOR_ID]
        self.assertTrue(np.all(camera_obs_surf[SENSORS[0]] == EXPECTED_STATES[0]))

        for i, item in enumerate(dataloader_surf):
            camera_obs_surf = item[AGENT_ID][SENSOR_ID]
            self.assertTrue(np.all(camera_obs_surf[SENSORS[0]] == EXPECTED_STATES[i]))
            if i >= DATASET_LEN - 1:
                break


if __name__ == "__main__":
    unittest.main()
