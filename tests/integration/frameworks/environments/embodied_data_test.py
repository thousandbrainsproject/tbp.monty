# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import quaternion as qt
from omegaconf import OmegaConf

from tbp.monty.experiment.environment import (
    Interface,
    OneObjectPerEpisodeInterface,
    OmniglotInterface,
    SaccadeOnImageFromStreamInterface,
    SaccadeOnImageInterface,
)
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.environments.object_init_samplers import (
    Default,
    MultiObjectNames,
)
from tbp.monty.frameworks.environments.two_d_data import (
    SaccadeOnImageEnvironment,
    SaccadeOnImageFromStreamEnvironment,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("agent_id_0")
SENSOR_ID = SensorID("sensor_id_0")
NUM_STEPS = 10
POSSIBLE_ACTIONS_DIST = [
    f"{AGENT_ID}.look_down",
    f"{AGENT_ID}.look_up",
    f"{AGENT_ID}.move_forward",
    f"{AGENT_ID}.turn_left",
    f"{AGENT_ID}.turn_right",
]
POSSIBLE_ACTIONS_ABS = [f"{AGENT_ID}.set_yaw", f"{AGENT_ID}.set_sensor_pitch"]
EXPECTED_STATES: npt.NDArray[np.uint8] = np.arange(0, NUM_STEPS, dtype=np.uint8)
TEST_DATALOADER_PATH = (
    Path(__file__).resolve().parents[3]
    / "unit"
    / "resources"
    / "dataloader_test_images"
)


class FakeEnvironmentRel(SimulatedObjectEnvironment):
    def __init__(self):
        self._current_state = 0

    @property
    def observations(self) -> Observations:
        return Observations(
            {
                AGENT_ID: AgentObservations(
                    {
                        SENSOR_ID: SensorObservation(
                            raw=EXPECTED_STATES[self._current_state]
                        )
                    }
                )
            }
        )

    @property
    def state(self) -> ProprioceptiveState:
        return ProprioceptiveState(
            {
                AGENT_ID: AgentState(
                    sensors={},
                    position=(0, 0, 0),
                    rotation=qt.quaternion(1, 0, 0, 0),
                )
            }
        )

    def add_object(
        self,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> ObjectInfo:
        return ObjectInfo(ObjectID(-1), SemanticID(-1))

    def step(
        self,
        actions: Sequence[Action],  # noqa: ARG002
    ) -> tuple[Observations, ProprioceptiveState]:
        self._current_state += 1
        return self.observations, self.state

    def remove_all_objects(self):
        pass

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._current_state = 0
        return self.observations, self.state

    def close(self):
        self._current_state = None


class FakeEnvironmentAbs(SimulatedObjectEnvironment):
    def __init__(self):
        self._current_state = 0

    def add_object(self, *_, **__) -> ObjectInfo:
        return ObjectInfo(ObjectID(-1), SemanticID(-1))

    @property
    def observations(self) -> Observations:
        return Observations(
            {
                AGENT_ID: AgentObservations(
                    {
                        SENSOR_ID: SensorObservation(
                            raw=EXPECTED_STATES[self._current_state]
                        )
                    }
                )
            }
        )

    @property
    def state(self) -> ProprioceptiveState:
        return ProprioceptiveState(
            {
                AGENT_ID: AgentState(
                    sensors={},
                    position=(0, 0, 0),
                    rotation=qt.quaternion(1, 0, 0, 0),
                )
            }
        )

    def step(
        self,
        actions: Sequence[Action],  # noqa: ARG002
    ) -> tuple[Observations, ProprioceptiveState]:
        self._current_state += 1
        return self.observations, self.state

    def remove_all_objects(self):
        pass

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._current_state = 0
        return self.observations, self.state

    def close(self):
        self._current_state = None


class FakeOmniglotEnvironment(FakeEnvironmentAbs):
    def __init__(self):
        super().__init__()
        self.alphabet_names = ["name_one", "name_two", "name_three"]


class EnvironmentInterfacePerObjectTest(unittest.TestCase):
    def test_accepts_plain_list_object_names(self):
        seed = 42
        rng = np.random.RandomState(seed)

        env_interface = OneObjectPerEpisodeInterface(
            env=FakeEnvironmentAbs(),
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
            object_names=["object_a", "object_b", "object_a"],
            object_init_sampler=Default(),
        )

        self.assertEqual(
            ["object_a", "object_b", "object_a"], env_interface.object_names
        )
        self.assertEqual(["object_a", "object_b"], env_interface.source_object_list)
        self.assertEqual(0, env_interface.num_distractors)

    def test_accepts_hydra_list_config_object_names(self):
        seed = 42
        rng = np.random.RandomState(seed)

        env_interface = OneObjectPerEpisodeInterface(
            env=FakeEnvironmentAbs(),
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
            object_names=OmegaConf.create(["object_a", "object_b", "object_a"]),
            object_init_sampler=Default(),
        )

        self.assertEqual(
            ["object_a", "object_b", "object_a"], env_interface.object_names
        )
        self.assertEqual(["object_a", "object_b"], env_interface.source_object_list)
        self.assertEqual(0, env_interface.num_distractors)

    def test_accepts_mapping_object_names(self):
        seed = 42
        rng = np.random.RandomState(seed)

        env_interface = OneObjectPerEpisodeInterface(
            env=FakeEnvironmentAbs(),
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
            object_names=MultiObjectNames(
                targets_list=["object_a", "object_c"],
                source_object_list=[
                    "object_a",
                    "object_b",
                    "object_c",
                    "object_b",
                ],
                num_distractors=2,
            ),
            object_init_sampler=Default(),
        )

        self.assertEqual(["object_a", "object_c"], env_interface.object_names)
        self.assertEqual(
            ["object_a", "object_b", "object_c"], env_interface.source_object_list
        )
        self.assertEqual(2, env_interface.num_distractors)

    def test_rejects_tuple_object_names(self):
        seed = 42
        rng = np.random.RandomState(seed)

        with self.assertRaisesRegex(
            TypeError,
            "Object names must be a list, ListConfig, or a mapping",
        ):
            OneObjectPerEpisodeInterface(
                env=FakeEnvironmentAbs(),
                rng=rng,
                seed=seed,
                experiment_mode=ExperimentMode.EVAL,
                object_names=("object_a", "object_b"),  # type: ignore[arg-type]
                object_init_sampler=Default(),
            )


class EmbodiedDataTest(unittest.TestCase):
    def test_embodied_env_interface_dist(self):
        seed = 42
        rng = np.random.RandomState(seed)
        env = FakeEnvironmentRel()
        env_interface_dist = Interface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        for i in range(1, NUM_STEPS):
            obs_dist, _ = env_interface_dist.step([])
            print(obs_dist)
            self.assertTrue(
                np.all(obs_dist[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[i])
            )

        initial_obs, _ = env_interface_dist.reset(rng)
        self.assertTrue(
            np.all(initial_obs[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[0])
        )
        obs_dist, _ = env_interface_dist.step([])
        self.assertFalse(
            np.all(
                obs_dist[AGENT_ID][SENSOR_ID]["raw"]
                == initial_obs[AGENT_ID][SENSOR_ID]["raw"]
            )
        )

    # @unittest.skip("debugging")
    def test_embodied_env_interface_abs(self):
        seed = 42
        rng = np.random.RandomState(seed)
        env = FakeEnvironmentAbs()
        env_interface_abs = Interface(
            env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        for i in range(1, NUM_STEPS):
            obs_abs, _ = env_interface_abs.step([])
            self.assertTrue(
                np.all(obs_abs[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[i])
            )

        initial_state, _ = env_interface_abs.reset(rng)
        self.assertTrue(
            np.all(initial_state[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[0])
        )
        obs_abs, _ = env_interface_abs.step([])
        self.assertFalse(
            np.all(
                obs_abs[AGENT_ID][SENSOR_ID]["raw"]
                == initial_state[AGENT_ID][SENSOR_ID]["raw"]
            )
        )

    # @unittest.skip("debugging")
    def test_embodied_env_interface_dist_states(self):
        seed = 42
        rng = np.random.RandomState(seed)

        env = FakeEnvironmentRel()
        env_interface_dist = Interface(
            env=env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Start at 1 because the initial call to reset consumes the zeroth state.
        i = 1
        while True:
            obs, _ = env_interface_dist.step([])
            self.assertTrue(
                np.all(obs[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[i])
            )
            if i >= NUM_STEPS - 1:
                break

            i += 1

    # @unittest.skip("debugging")
    def test_embodied_env_interface_abs_states(self):
        seed = 42
        rng = np.random.RandomState(seed)

        env = FakeEnvironmentAbs()
        env_interface_abs = Interface(
            env=env,
            rng=rng,
            seed=seed,
            experiment_mode=ExperimentMode.EVAL,
        )

        # Start at 1 because the initial call to reset consumes the zeroth state.
        i = 1
        while True:
            obs, _ = env_interface_abs.step([])
            self.assertTrue(
                np.all(obs[AGENT_ID][SENSOR_ID]["raw"] == EXPECTED_STATES[i])
            )
            if i >= NUM_STEPS - 1:
                break

            i += 1

    def check_two_d_patch_obs(
        self,
        obs: SensorObservation,
        patch_size: int,
        expected_keys: Sequence[str],
    ):
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

        alphabets = [0, 0, 0, 1, 1, 1]
        characters = [1, 2, 3, 1, 2, 3]
        versions = [1, 1, 1, 1, 1, 1]
        num_objects = len(characters)

        env = FakeOmniglotEnvironment()
        omniglot_data_loader_abs = OmniglotInterface(
            env=env,  # TODO: FakeOmniglotEnvironment is not an OmniglotEnvironment
            rng=rng,
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
        sensor_id = SensorID("patch")
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = TEST_DATALOADER_PATH

        env_init_args = {"patch_size": patch_size, "data_path": data_path}
        env = SaccadeOnImageEnvironment(**env_init_args)
        env_interface_rel = SaccadeOnImageInterface(
            env=env,
            rng=rng,
            scenes=[0, 0],
            versions=[0, 1],
        )
        env_interface_rel.pre_episode(rng)
        initial_state, _ = env_interface_rel.step([])
        sensed_data = initial_state[AGENT_ID][sensor_id]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        i = 0
        while True:
            obs, _ = env_interface_rel.step([])
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

            i += 1

        env_interface_rel.post_episode()
        self.assertEqual(
            env_interface_rel.env.scene_version,
            1,
            "Did not cycle to next scene version.",
        )
        env_interface_rel.pre_episode(rng)
        i = 0
        while True:
            obs, _ = env_interface_rel.step([])
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

            i += 1

    def test_saccade_on_image_stream_env_interface(self):
        rng = np.random.RandomState(42)
        sensor_id = SensorID("patch")
        patch_size = 48
        expected_keys = ["depth", "rgba", "pixel_loc"]

        data_path = TEST_DATALOADER_PATH / "0_numenta_mug"

        env_init_args = {"patch_size": patch_size, "data_path": data_path}
        env = SaccadeOnImageFromStreamEnvironment(**env_init_args)
        env_interface_rel = SaccadeOnImageFromStreamInterface(env=env, rng=rng)
        env_interface_rel.pre_episode(rng)
        initial_state, _ = env_interface_rel.step([])
        sensed_data = initial_state[AGENT_ID][sensor_id]
        self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)

        i = 0
        while True:
            obs, _ = env_interface_rel.step([])
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

            i += 1

        env_interface_rel.post_episode()
        self.assertEqual(
            env_interface_rel.env.current_scene,
            1,
            "Did not cycle to next scene version.",
        )
        i = 0
        while True:
            obs, _ = env_interface_rel.step([])
            sensed_data = obs[AGENT_ID][sensor_id]
            self.check_two_d_patch_obs(sensed_data, patch_size, expected_keys)
            if i >= NUM_STEPS - 1:
                break

            i += 1
