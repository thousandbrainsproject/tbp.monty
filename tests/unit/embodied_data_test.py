# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import random
import unittest
from pathlib import Path

import numpy as np

from tbp.monty.frameworks.actions.action_samplers import (
    UniformlyDistributedSampler,
)
from tbp.monty.frameworks.config_utils.config_args import make_base_policy_config
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
    EnvironmentDataset,
    SaccadeOnImageDataLoader,
    SaccadeOnImageFromStreamDataLoader,
)
from tbp.monty.frameworks.environments.embodied_environment import (
    ActionSpace,
    EmbodiedEnvironment,
)
from tbp.monty.frameworks.environments.two_d_data import (
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem

AGENT_ID = "agent_id_0"
SENSOR_ID = "sensor_id_0"
DATASET_LEN = 10
POSSIBLE_ACTIONS_DIST = [
    f"{AGENT_ID}.look_down",
    f"{AGENT_ID}.look_up",
    f"{AGENT_ID}.move_forward",
    f"{AGENT_ID}.turn_left",
    f"{AGENT_ID}.turn_right",
]
POSSIBLE_ACTIONS_ABS = [f"{AGENT_ID}.set_yaw", f"{AGENT_ID}.set_sensor_pitch"]
EXPECTED_ACTIONS_DIST = [
    POSSIBLE_ACTIONS_DIST[i]
    for i in np.random.randint(0, len(POSSIBLE_ACTIONS_DIST), 100)
]
EXPECTED_ACTIONS_ABS = [
    POSSIBLE_ACTIONS_ABS[i]
    for i in np.random.randint(0, len(POSSIBLE_ACTIONS_ABS), 100)
]
EXPECTED_STATES = np.random.rand(DATASET_LEN)


class FakeActionSpace(tuple, ActionSpace):
    def sample(self):
        return random.choice(self)


class FakeEnvironmentRel(EmbodiedEnvironment):
    def __init__(self):
        self._current_state = 0

    @property
    def action_space(self):
        return FakeActionSpace(EXPECTED_ACTIONS_DIST)

    def add_object(self, *args, **kwargs):
        return None

    def step(self, action):
        self._current_state += 1
        obs = {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }
        return obs

    def get_state(self):
        return None

    def remove_all_objects(self):
        pass

    def reset(self):
        self._current_state = 0
        obs = {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }
        return obs

    def close(self):
        self._current_state = None


class FakeEnvironmentAbs(EmbodiedEnvironment):
    def __init__(self):
        self._current_state = 0

    @property
    def action_space(self):
        return FakeActionSpace(EXPECTED_ACTIONS_ABS)

    def add_object(self, *args, **kwargs):
        return None

    def step(self, action):
        self._current_state += 1
        obs = {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }
        return obs

    def get_state(self):
        return None

    def remove_all_objects(self):
        pass

    def reset(self):
        self._current_state = 0
        obs = {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }
        return obs

    def close(self):
        self._current_state = None


class EmbodiedDataTest(unittest.TestCase):
    def test_embodied_dataset_dist(self):
        rng = np.random.RandomState(42)
        dataset_dist = EnvironmentDataset(
            env_init_func=FakeEnvironmentRel, env_init_args={}, rng=rng
        )

        action_space_dist = dataset_dist.action_space
        self.assertIsInstance(action_space_dist, ActionSpace)
        self.assertSequenceEqual(action_space_dist, EXPECTED_ACTIONS_DIST)
        self.assertIn(action_space_dist.sample(), EXPECTED_ACTIONS_DIST)

        base_policy_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_dist.__dict__)
        )

        for i in range(1, DATASET_LEN):
            obs_dist, _ = dataset_dist[motor_system_dist()]
            print(obs_dist)
            self.assertTrue(
                np.all(obs_dist[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )

        initial_state, _ = dataset_dist.reset()
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )
        obs_dist, _ = dataset_dist[motor_system_dist()]
        self.assertFalse(
            np.all(
                obs_dist[AGENT_ID][SENSOR_ID]["sensor"]
                == initial_state[AGENT_ID][SENSOR_ID]["sensor"]
            )
        )

        dataset_dist.close()
        with self.assertRaises(Exception):  # noqa: B017
            obs_dist, _ = dataset_dist[motor_system_dist()]

    # @unittest.skip("debugging")
    def test_embodied_dataset_abs(self):
        rng = np.random.RandomState(42)

        dataset_abs = EnvironmentDataset(
            env_init_func=FakeEnvironmentAbs, env_init_args={}, rng=rng
        )

        action_space_abs = dataset_abs.action_space
        self.assertIsInstance(action_space_abs, ActionSpace)
        self.assertSequenceEqual(action_space_abs, EXPECTED_ACTIONS_ABS)
        self.assertIn(action_space_abs.sample(), EXPECTED_ACTIONS_ABS)

        base_policy_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_abs.__dict__)
        )

        for i in range(1, DATASET_LEN):
            obs_abs, _ = dataset_abs[motor_system_abs()]
            self.assertTrue(
                np.all(obs_abs[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )

        initial_state, _ = dataset_abs.reset()
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )
        obs_abs, _ = dataset_abs[motor_system_abs()]
        self.assertFalse(
            np.all(
                obs_abs[AGENT_ID][SENSOR_ID]["sensor"]
                == initial_state[AGENT_ID][SENSOR_ID]["sensor"]
            )
        )

        dataset_abs.close()
        with self.assertRaises(Exception):  # noqa: B017
            obs_abs, _ = dataset_abs[motor_system_abs()]

    # @unittest.skip("debugging")
    def test_embodied_dataloader_dist(self):
        rng = np.random.RandomState(42)
        dataset_dist = EnvironmentDataset(
            env_init_func=FakeEnvironmentRel,
            env_init_args={},
            rng=rng,
        )

        base_policy_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_dist.__dict__)
        )

        dataloader_dist = EnvironmentDataLoader(
            dataset_dist, motor_system_dist, rng=rng
        )
        initial_state = next(dataloader_dist)
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )

        for i, item in enumerate(dataloader_dist):
            self.assertTrue(
                np.all(item[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )
            if i >= DATASET_LEN - 1:
                break

    # @unittest.skip("debugging")
    def test_embodied_dataloader_abs(self):
        rng = np.random.RandomState(42)

        dataset_abs = EnvironmentDataset(
            env_init_func=FakeEnvironmentAbs, env_init_args={}, rng=rng
        )

        base_policy_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_abs.__dict__)
        )

        dataloader_abs = EnvironmentDataLoader(dataset_abs, motor_system_abs, rng)
        initial_state = next(dataloader_abs)
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )

        for i, item in enumerate(dataloader_abs):
            self.assertTrue(
                np.all(item[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )
            if i >= DATASET_LEN - 1:
                break

    def check_two_d_patch_obs(self, obs, patch_size, expected_keys):
        for key in expected_keys:
            self.assertIn(
                key,
                obs.keys(),
                f"{key} not in sensor data {obs.keys()}",
            )
        depth_obs_size = obs["depth"].shape
        self.assertEqual(
            depth_obs_size,
            (patch_size, patch_size),
            f"extracted patch of size {depth_obs_size} does not have specified "
            f"patch size {patch_size} along x dimension",
        )
        rgba_obs_size = obs["rgba"].shape
        self.assertEqual(
            rgba_obs_size,
            (patch_size, patch_size, 4),
            f"extracted patch of size {rgba_obs_size} does not have specified "
            f"patch size {patch_size}",
        )

    def check_two_d_movement(self, prev_loc, current_loc, action):
        if action != AGENT_ID + ".move_forward":
            self.assertGreater(
                np.linalg.norm(prev_loc - current_loc),
                0.0001,
                "Agent did not move",
            )
        else:
            self.assertLess(
                np.linalg.norm(prev_loc - current_loc),
                0.0001,
                "Agent should not have moved",
            )

    def test_saccade_on_image_dataloader(self):
        rng = np.random.RandomState(42)
        sensor_id = "patch"
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = "./resources/dataloader_test_images/"
        data_path = os.path.join(
            Path(__file__).parent, "resources/dataloader_test_images/"
        )

        dataset_rel = EnvironmentDataset(
            env_init_func=SaccadeOnImageEnvironment,
            env_init_args={"patch_size": patch_size, "data_path": data_path},
            rng=rng,
        )

        base_policy_config_rel = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )

        motor_system_rel = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_rel.__dict__)
        )

        dataloader_rel = SaccadeOnImageDataLoader(
            scenes=[0, 0],
            versions=[0, 1],
            dataset=dataset_rel,
            motor_system=motor_system_rel,
            rng=rng,
        )
        dataloader_rel.pre_episode()
        initial_state = next(dataloader_rel)
        sensed_data = initial_state[AGENT_ID][sensor_id]
        current_state = dataloader_rel.dataset.env.get_state()
        prev_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"]["position"]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        for i, obs in enumerate(dataloader_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            current_state = dataloader_rel.dataset.env.get_state()
            current_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"][
                "position"
            ]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            action = dataloader_rel._action
            action_name = f"{action.agent_id}.{action.name}"
            self.check_two_d_movement(prev_loc, current_loc, action_name)
            prev_loc = current_loc
            if i >= DATASET_LEN - 1:
                break

        dataloader_rel.post_episode()
        self.assertEqual(
            dataloader_rel.dataset.env.scene_version,
            1,
            "Did not cycle to next scene version.",
        )
        dataloader_rel.pre_episode()
        for i, obs in enumerate(dataloader_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            current_state = dataloader_rel.dataset.env.get_state()
            current_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"][
                "position"
            ]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            action = dataloader_rel._action
            action_name = (
                f"{action.agent_id}.{action.name}" if action is not None else None
            )
            self.check_two_d_movement(prev_loc, current_loc, action_name)
            prev_loc = current_loc
            if i >= DATASET_LEN - 1:
                break

    def test_saccade_on_image_stream_dataloader(self):
        rng = np.random.RandomState(42)
        sensor_id = "patch"
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = "./resources/dataloader_test_images/"
        data_path = os.path.join(
            Path(__file__).parent,
            "resources/dataloader_test_images/0_numenta_mug/",
        )

        dataset_rel = EnvironmentDataset(
            env_init_func=SaccadeOnImageFromStreamEnvironment,
            env_init_args={"patch_size": patch_size, "data_path": data_path},
            rng=rng,
        )

        base_policy_config_rel = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )

        motor_system_rel = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_rel.__dict__)
        )

        dataloader_rel = SaccadeOnImageFromStreamDataLoader(
            dataset=dataset_rel,
            motor_system=motor_system_rel,
            rng=rng,
        )
        dataloader_rel.pre_episode()
        initial_state = next(dataloader_rel)
        sensed_data = initial_state[AGENT_ID][sensor_id]
        current_state = dataloader_rel.dataset.env.get_state()
        prev_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"]["position"]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        for i, obs in enumerate(dataloader_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            current_state = dataloader_rel.dataset.env.get_state()
            current_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"][
                "position"
            ]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            action = dataloader_rel._action
            action_name = f"{action.agent_id}.{action.name}"
            self.check_two_d_movement(prev_loc, current_loc, action_name)
            prev_loc = current_loc
            if i >= DATASET_LEN - 1:
                break
        dataloader_rel.post_episode()
        self.assertEqual(
            dataloader_rel.dataset.env.current_scene,
            1,
            "Did not cycle to next scene version.",
        )
        for i, obs in enumerate(dataloader_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            current_state = dataloader_rel.dataset.env.get_state()
            current_loc = current_state[AGENT_ID]["sensors"][sensor_id + ".depth"][
                "position"
            ]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            action = dataloader_rel._action
            action_name = f"{action.agent_id}.{action.name}"
            self.check_two_d_movement(prev_loc, current_loc, action_name)
            prev_loc = current_loc
            if i >= DATASET_LEN - 1:
                break


if __name__ == "__main__":
    unittest.main()
