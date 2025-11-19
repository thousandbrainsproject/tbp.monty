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
import unittest
from pathlib import Path

import numpy as np
from typing_extensions import override

from tbp.monty.frameworks.actions.action_samplers import (
    UniformlyDistributedSampler,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.config_utils.config_args import make_base_policy_config
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterface,
    OmniglotEnvironmentInterface,
    SaccadeOnImageEnvironmentInterface,
    SaccadeOnImageFromStreamEnvironmentInterface,
)
from tbp.monty.frameworks.environments.embodied_environment import (
    EmbodiedEnvironment,
    ObjectID,
)
from tbp.monty.frameworks.environments.two_d_data import (
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem

AGENT_ID = AgentID("agent_id_0")
SENSOR_ID = "sensor_id_0"
NUM_STEPS = 10
POSSIBLE_ACTIONS_DIST = [
    f"{AGENT_ID}.look_down",
    f"{AGENT_ID}.look_up",
    f"{AGENT_ID}.move_forward",
    f"{AGENT_ID}.turn_left",
    f"{AGENT_ID}.turn_right",
]
POSSIBLE_ACTIONS_ABS = [f"{AGENT_ID}.set_yaw", f"{AGENT_ID}.set_sensor_pitch"]
EXPECTED_STATES = np.random.rand(NUM_STEPS)


class FakeEnvironmentRel(EmbodiedEnvironment):
    def __init__(self):
        self._current_state = 0

    @override
    def add_object(self, *args, **kwargs) -> ObjectID:
        return ObjectID(-1)

    @override
    def step(self, actions):
        self._current_state += 1
        return {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }

    def get_state(self):
        return None

    def remove_all_objects(self):
        pass

    def reset(self):
        self._current_state = 0
        return {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }

    def close(self):
        self._current_state = None


class FakeEnvironmentAbs(EmbodiedEnvironment):
    def __init__(self):
        self._current_state = 0

    @override
    def add_object(self, *args, **kwargs) -> ObjectID:
        return ObjectID(-1)

    @override
    def step(self, actions):
        self._current_state += 1
        return {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }

    def get_state(self):
        return None

    def remove_all_objects(self):
        pass

    def reset(self):
        self._current_state = 0
        return {
            f"{AGENT_ID}": {
                f"{SENSOR_ID}": {"sensor": EXPECTED_STATES[self._current_state]}
            }
        }

    def close(self):
        self._current_state = None


class FakeOmniglotEnvironment(FakeEnvironmentAbs):
    def __init__(self):
        self.alphabet_names = ["name_one", "name_two", "name_three"]


class EmbodiedDataTest(unittest.TestCase):
    def test_embodied_env_interface_dist(self):
        rng = np.random.RandomState(42)
        base_policy_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_dist.__dict__)
        )
        env = FakeEnvironmentRel()
        env_interface_dist = EnvironmentInterface(
            env,
            rng=rng,
            motor_system=motor_system_dist,
        )

        for i in range(1, NUM_STEPS):
            obs_dist, _ = env_interface_dist.step(motor_system_dist())
            print(obs_dist)
            self.assertTrue(
                np.all(obs_dist[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )

        initial_state, _ = env_interface_dist.reset()
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )
        obs_dist, _ = env_interface_dist.step(motor_system_dist())
        self.assertFalse(
            np.all(
                obs_dist[AGENT_ID][SENSOR_ID]["sensor"]
                == initial_state[AGENT_ID][SENSOR_ID]["sensor"]
            )
        )

    # @unittest.skip("debugging")
    def test_embodied_env_interface_abs(self):
        rng = np.random.RandomState(42)

        base_policy_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_abs.__dict__)
        )
        env = FakeEnvironmentAbs()
        env_interface_abs = EnvironmentInterface(
            env,
            rng=rng,
            motor_system=motor_system_abs,
        )

        for i in range(1, NUM_STEPS):
            obs_abs, _ = env_interface_abs.step(motor_system_abs())
            self.assertTrue(
                np.all(obs_abs[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )

        initial_state, _ = env_interface_abs.reset()
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[0])
        )
        obs_abs, _ = env_interface_abs.step(motor_system_abs())
        self.assertFalse(
            np.all(
                obs_abs[AGENT_ID][SENSOR_ID]["sensor"]
                == initial_state[AGENT_ID][SENSOR_ID]["sensor"]
            )
        )

    # @unittest.skip("debugging")
    def test_embodied_env_interface_dist_states(self):
        rng = np.random.RandomState(42)
        base_policy_config_dist = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_dist = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_dist.__dict__)
        )
        env = FakeEnvironmentRel()
        env_interface_dist = EnvironmentInterface(
            env=env, rng=rng, motor_system=motor_system_dist
        )

        for i, item in enumerate(env_interface_dist):
            self.assertTrue(
                np.all(item[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )
            if i >= NUM_STEPS - 1:
                break

    # @unittest.skip("debugging")
    def test_embodied_env_interface_abs_states(self):
        rng = np.random.RandomState(42)

        base_policy_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_abs.__dict__)
        )
        env = FakeEnvironmentAbs()
        env_interface_abs = EnvironmentInterface(
            env=env, rng=rng, motor_system=motor_system_abs
        )

        for i, item in enumerate(env_interface_abs):
            self.assertTrue(
                np.all(item[AGENT_ID][SENSOR_ID]["sensor"] == EXPECTED_STATES[i])
            )
            if i >= NUM_STEPS - 1:
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

    def test_omniglot_data_loader(self):
        rng = np.random.RandomState(42)

        base_policy_config_abs = make_base_policy_config(
            action_space_type="absolute_only",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )
        motor_system_abs = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_abs.__dict__)
        )

        alphabets = [0, 0, 0, 1, 1, 1]
        characters = [1, 2, 3, 1, 2, 3]
        versions = [1, 1, 1, 1, 1, 1]
        num_objects = len(characters)

        env = FakeOmniglotEnvironment()
        omniglot_data_loader_abs = OmniglotEnvironmentInterface(
            env=env,
            rng=rng,
            motor_system=motor_system_abs,
            alphabets=alphabets,
            characters=characters,
            versions=versions,
        )

        self.assertEqual(
            alphabets,
            omniglot_data_loader_abs.alphabets,
            "Env not initiated.",
        )
        self.assertEqual("name_one_1", omniglot_data_loader_abs.object_names[0])
        self.assertEqual(num_objects, omniglot_data_loader_abs.n_objects)

    def test_saccade_on_image_env_interface(self):
        rng = np.random.RandomState(42)
        sensor_id = "patch"
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = "./resources/dataloader_test_images/"
        data_path = os.path.join(
            Path(__file__).parent, "resources/dataloader_test_images/"
        )

        base_policy_config_rel = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )

        motor_system_rel = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_rel.__dict__)
        )

        env_init_args = {"patch_size": patch_size, "data_path": data_path}
        env = SaccadeOnImageEnvironment(**env_init_args)
        env_interface_rel = SaccadeOnImageEnvironmentInterface(
            env=env,
            rng=rng,
            motor_system=motor_system_rel,
            scenes=[0, 0],
            versions=[0, 1],
        )
        env_interface_rel.pre_episode()
        initial_state = next(env_interface_rel)
        sensed_data = initial_state[AGENT_ID][sensor_id]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        for i, obs in enumerate(env_interface_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

        env_interface_rel.post_episode()
        self.assertEqual(
            env_interface_rel.env.scene_version,
            1,
            "Did not cycle to next scene version.",
        )
        env_interface_rel.pre_episode()
        for i, obs in enumerate(env_interface_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

    def test_saccade_on_image_stream_env_interface(self):
        rng = np.random.RandomState(42)
        sensor_id = "patch"
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = "./resources/dataloader_test_images/"
        data_path = os.path.join(
            Path(__file__).parent,
            "resources/dataloader_test_images/0_numenta_mug/",
        )

        base_policy_config_rel = make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
            agent_id=AGENT_ID,
        )

        motor_system_rel = MotorSystem(
            policy=BasePolicy(rng=rng, **base_policy_config_rel.__dict__)
        )

        env_init_args = {"patch_size": patch_size, "data_path": data_path}
        env = SaccadeOnImageFromStreamEnvironment(**env_init_args)
        env_interface_rel = SaccadeOnImageFromStreamEnvironmentInterface(
            env=env, rng=rng, motor_system=motor_system_rel
        )
        env_interface_rel.pre_episode()
        initial_state = next(env_interface_rel)
        sensed_data = initial_state[AGENT_ID][sensor_id]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        for i, obs in enumerate(env_interface_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

        env_interface_rel.post_episode()
        self.assertEqual(
            env_interface_rel.env.current_scene,
            1,
            "Did not cycle to next scene version.",
        )
        for i, obs in enumerate(env_interface_rel):
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break


if __name__ == "__main__":
    unittest.main()
