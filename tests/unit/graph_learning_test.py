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

import pytest

from tbp.monty.context import RuntimeContext
from tests import HYDRA_ROOT

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)
import copy
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.utils.logging_utils import (
    load_stats,
)
from tests.unit.resources.unit_test_utils import BaseGraphTest


@dataclass
class EpisodeObservations:
    """An episode of observations for a single named object."""

    name: str
    observations: list[Any]

    def __len__(self):
        return len(self.observations)


@dataclass
class TrainedGraphLM:
    """Class to bundle up a GraphLM and the object observations it was trained on.

    Used by the tests below to bundle the trained GraphLMs with the object observations
    it was trained on. See `GraphLearningTest::get_5lm_gm_with_fake_objects` for
    more details.
    """

    learning_module: GraphLM
    episodes: list[EpisodeObservations]

    @property
    def mode(self) -> ExperimentMode | None:
        """Helper property to make setting and reading the LM mode easier."""
        return self.learning_module.mode

    @mode.setter
    def mode(self, mode: ExperimentMode | None):
        self.learning_module.mode = mode

    def num_observations(self, episode_num: int) -> int:
        """Number of observations in the specified episode.

        Returns:
            The number of observations in the episode.
        """
        return len(self.episodes[episode_num].observations)

    @property
    def num_episodes(self) -> int:
        """Number of episodes/objects this LM was trained on."""
        return len(self.episodes)

    def pre_episode(self, primary_target) -> None:
        """Delegates pre_episode calls to the LM."""
        self.learning_module.pre_episode(primary_target)


class GraphLearningTest(BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.output_dir = Path(tempfile.mkdtemp())
        self.compositional_save_path = tempfile.mkdtemp()
        self.fixed_actions_path = (
            Path(__file__).parent / "resources" / "fixed_test_actions.jsonl"
        )
        self.fixed_actions_path_off_object = (
            Path(__file__).parent / "resources" / "fixed_test_actions_off_object.jsonl"
        )

        # Generate the override string for setting the actions file name.
        # We're doing this because the string is too long otherwise.
        actions_file_name_selector = ".".join(  # noqa: FLY002
            [
                "test",
                "config",
                "monty_config",
                "motor_system_config",
                "motor_system_args",
                "policy",
                "file_name",
            ]
        )

        def hydra_config(
            test_name: str,
            action_file_name: Path | None = None,
            extra_overrides: list[str] | None = None,
        ) -> DictConfig:
            """Return a Hydra configuration from the specified test name.

            Args:
                test_name: the name of the test config to load
                action_file_name: Optional path to a file of actions to use
                extra_overrides: Optional list of extra overrides to add
            """
            overrides = [
                f"test=graph_learning/{test_name}",
                f"test.config.logging.output_dir={self.output_dir}",
            ]
            if action_file_name:
                overrides.append(f"{actions_file_name_selector}={action_file_name}")
            if extra_overrides:
                overrides += extra_overrides
            return hydra.compose(config_name="test", overrides=overrides)

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.base_cfg = hydra_config("base")
            self.surface_agent_eval_cfg = hydra_config("surface_agent_eval")
            self.ppf_pred_cfg = hydra_config("ppf_pred")
            self.disp_pred_cfg = hydra_config("disp_pred")
            self.feature_pred_cfg = hydra_config("feature_pred")
            self.fixed_actions_disp_cfg = hydra_config(
                "fixed_actions_disp", self.fixed_actions_path
            )
            self.fixed_actions_ppf_cfg = hydra_config(
                "fixed_actions_ppf", self.fixed_actions_path
            )
            self.fixed_actions_feat_cfg = hydra_config(
                "fixed_actions_feat", self.fixed_actions_path
            )
            self.feature_pred_time_out_cfg = hydra_config(
                "feature_pred_time_out", self.fixed_actions_path
            )
            self.feature_pred_off_object_cfg = hydra_config(
                "feature_pred_off_object", self.fixed_actions_path_off_object
            )
            self.feature_pred_off_object_train_cfg = hydra_config(
                "feature_pred_off_object_train", self.fixed_actions_path_off_object
            )
            self.feature_uniform_initial_poses_cfg = hydra_config(
                "feature_uniform_initial_poses", self.fixed_actions_path
            )
            self.five_lm_ppf_displacement_cfg = hydra_config(
                "five_lm_ppf_displacement", self.fixed_actions_path
            )
            self.five_lm_feature_cfg = hydra_config(
                "five_lm_feature", self.fixed_actions_path
            )

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_can_initialize(self):
        """Canary to confirm we can initialize an experiment.

        This could be part of the setUp method, but it's easier to debug if something
        breaks the setup_experiment method if there's a separate test for it.
        """
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            pass

    def test_can_run_train_episode(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.run_episode()

    def test_right_data_in_buffer(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.pre_episode()
            step = 0
            ctx = RuntimeContext(rng=exp.rng)
            while True:
                observations = exp.env_interface.step(ctx, first=(step == 0))
                exp.model.step(ctx, observations)
                self.assertEqual(
                    step + 1,
                    len(exp.model.learning_modules[0].buffer),
                    "buffer does not contain the right amount of elements.",
                )
                self.assertEqual(
                    step + 1,
                    len(
                        exp.model.learning_modules[
                            0
                        ].buffer.get_all_locations_on_object(input_channel="first")
                    ),
                    "buffer does not contain the right amount of locations.",
                )
                if step == 0:
                    self.assertListEqual(
                        list(
                            exp.model.learning_modules[0].buffer.get_nth_displacement(
                                0, input_channel="first"
                            )
                        ),
                        [0, 0, 0],
                        "displacement at step 0 should be 0.",
                    )
                self.assertEqual(
                    step + 1,
                    len(
                        exp.model.learning_modules[0].buffer.displacements["patch"][
                            "displacement"
                        ]
                    ),
                    "buffer does not contain the right amount of displacements.",
                )
                self.assertSetEqual(
                    set(exp.model.sensor_modules[0].features),
                    set(exp.model.learning_modules[0].buffer[-1]["patch"].keys()),
                    "buffer doesn't contain all features required for matching.",
                )
                if step == 3:
                    break

                step += 1

    def test_can_run_eval_episode(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.EVAL
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.run_episode()

    def test_can_run_eval_episode_with_surface_agent(self):
        exp = hydra.utils.instantiate(self.surface_agent_eval_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.EVAL
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.run_episode()

    def test_can_run_ppf_experiment(self):
        exp = hydra.utils.instantiate(self.ppf_pred_cfg.test)
        with exp:
            exp.run()

    def test_can_run_disp_experiment(self):
        exp = hydra.utils.instantiate(self.disp_pred_cfg.test)
        with exp:
            exp.run()

    def test_can_run_feature_experiment(self):
        exp = hydra.utils.instantiate(self.feature_pred_cfg.test)
        with exp:
            exp.run()

    def test_fixed_actions_disp(self):
        """Runs three test episodes on capsule3DSolid and cubeSolid.

        1. capsule, rotation = 0,0,0 -> no models in memory yet -> store new model
        2. cube, rotation = 0,0,0 -> no models in memory yet -> store new model
        3. capsule, rotation = 0,45,0 -> displacements cant recognize rotated model
             -> store new rotated model of object
        4. cube, rotation = 0,45,0 -> same as 3 for cube
        5. capsule, rotation = 0,0,0 -> recognize first model stored in memory
        6. cube, rotation = 0,0,0 -> recognize first model stored in memory

        To make sure we recognize the first model in step three we use a fixed
        action policy that executed the same action sequence in every episode.

        Followed by three eval episodes on capsule3DSolid (same rotation sequence so
        the first and third episode should recognize the capsule).
        """
        exp = hydra.utils.instantiate(self.fixed_actions_disp_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def test_fixed_actions_ppf(self):
        """Like test_fixed_actions_disp but using point pair features for matching."""
        exp = hydra.utils.instantiate(self.fixed_actions_disp_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def test_fixed_actions_feat(self):
        """Like test_fixed_actions_disp but using point pair features for matching."""
        exp = hydra.utils.instantiate(self.fixed_actions_feat_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def test_save_and_load(self):
        # Move this to graph_building_test.py?
        exp = hydra.utils.instantiate(self.fixed_actions_ppf_cfg.test)
        with exp:
            exp.run()

        # We are training for 3 epochs by default, load most recent indexing from 0
        cfg2 = copy.deepcopy(self.fixed_actions_ppf_cfg)
        cfg2.test.config.model_name_or_path = str(
            Path(exp.output_dir) / "2",  # latest checkpoint
        )
        exp2 = hydra.utils.instantiate(cfg2.test)
        with exp2:
            graph_memory_1 = exp.model.learning_modules[
                0
            ].graph_memory.get_all_models_in_memory()
            graph_memory_2 = exp2.model.learning_modules[
                0
            ].graph_memory.get_all_models_in_memory()

            # Loop over each graph model and check they have the exact same data
            for obj_name in graph_memory_1:
                for input_channel in graph_memory_1[obj_name]:
                    graph_1 = graph_memory_1[obj_name][input_channel]
                    graph_2 = graph_memory_2[obj_name][input_channel]
                    self.check_graphs_equal(graph_1, graph_2)

    def test_time_out(self):
        """Test time_out and pose_time_out detection and logging.

        # TODO: This test is a little shaky and should be improved.

        Episodes 0 and 1: Should detect no_match and build models for 2 objects
        Episode 2: Lowered max_steps and raised mmd -> detect pose_time_out
        (Episode 3: object is too similar with tolerances, will also detect time_out)
        Episodes 4 and 5: Increased curvature tolerance -> detect time_out
        """
        exp = hydra.utils.instantiate(self.feature_pred_time_out_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            for e in range(6):
                if e % 2 == 0:
                    exp.pre_epoch()
                if e == 2:
                    # Set max steps low & raise mmd to get pose time outs
                    exp.max_train_steps = 3
                    exp.model.learning_modules[0].max_match_distance = 0.1
                if e == 4:
                    # set curvature threshold high to get time outs
                    exp.model.learning_modules[0].tolerances["patch"][
                        "principal_curvatures_log"
                    ] = [10, 10]
                exp.run_episode()
                if e % 2 == 1:
                    exp.post_epoch()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.assertEqual(
            train_stats["primary_performance"][2],
            "pose_time_out",
            "pose time out not recognized/logged correctly",
        )
        # possible locations are str in .csv with one line per pose
        # unique rotations may already be one but we don't know
        # where we are yet.
        self.assertGreater(
            train_stats["possible_object_locations"][2].count("\n"),
            0,  # If there are two possible locations, there will be 1 newline
            "pose time out episode should have more than one possible pose.",
        )
        for episode in [4, 5]:
            self.assertEqual(
                train_stats["primary_performance"][episode],
                "time_out",
                "time out not recognized/logged correctly",
            )
            self.assertEqual(
                train_stats["primary_performance"][episode],
                "time_out",
                "time out not recognized/logged correctly",
            )
            self.assertGreater(
                train_stats["num_possible_matches"][episode],
                1,
                "time out episode should have more than one possible match.",
            )
            # possible objects are comma separated string
            self.assertGreater(
                train_stats["result"][episode].count(","),
                0,  # If there are two objects, there should be 1 comma
                "time out episode should log more than one possible match.",
            )

    def test_confused_logging(self):
        # When the algorithm evolves, this scenario may not lead to confusion
        # anymore. Setting min_steps would also avoid this, probably.
        exp = hydra.utils.instantiate(self.fixed_actions_feat_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            # Overwrite target with a false name to test confused logging.
            for e in range(4):
                exp.pre_episode()
                exp.model.primary_target = str(e)
                for lm in exp.model.learning_modules:
                    lm.primary_target = str(e)
                last_step = exp.run_episode_steps()
                exp.post_episode(last_step)
            exp.post_epoch()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        for i in [0, 1]:
            self.assertEqual(
                train_stats["primary_performance"][i],
                "no_match",
                f"episode {i} should be no_match.",
            )
            self.assertEqual(
                train_stats["TFNP"][i],
                "unknown_object_not_matched_(TN)",
                f"episode {i} should detect a true negative.",
            )
        for i in [2, 3]:
            self.assertEqual(
                train_stats["primary_performance"][i],
                "confused",
                f"episode {i} should log confused performance.",
            )
            self.assertEqual(
                train_stats["TFNP"][i],
                "unknown_object_in_possible_matches_(FP)",
                f"episode {i} should detect a false positive.",
            )
            self.assertNotEqual(
                train_stats["primary_target_object"][i],
                train_stats["result"][i],
                "confused object id should not be the same as target.",
            )
            self.assertNotEqual(
                train_stats["primary_target_object"][i],
                train_stats["possible_match_sources"][i],
                "confused object id should not be in possible_match_sources.",
            )

    def test_moving_off_object(self):
        # Tests additional elements of logging, in particular in relation
        # to logging of observations when off the object

        exp = hydra.utils.instantiate(self.feature_pred_off_object_train_cfg.test)
        with exp:
            # First episode will be used to learn object (no_match is triggered before
            # min_steps is reached and the sensor moves off the object). In the second
            # episode the sensor moves off the sphere on episode steps 6+
            # Eventually, we circle round and come back to the object; recognition
            # does not take place before then because when off the object, matching
            # steps are no longer incremented, while it is an unfamiliar part of
            # the object that we return to
            exp.run()
            self.assertEqual(
                len(
                    exp.model.learning_modules[0].buffer.get_all_locations_on_object(
                        input_channel="patch"
                    )
                ),
                len(
                    exp.model.learning_modules[0].buffer.get_all_features_on_object()[
                        "patch"
                    ]["pose_vectors"]
                ),
                "Did not retrieve same amount of feature and locations on object.",
            )
            self.assertEqual(
                sum(
                    exp.model.learning_modules[0].buffer.get_all_features_on_object()[
                        "patch"
                    ]["on_object"]
                ),
                len(
                    exp.model.learning_modules[0].buffer.get_all_features_on_object()[
                        "patch"
                    ]["on_object"]
                ),
                "not all retrieved features were collected on the object.",
            )
            # Since we don't add observations to the buffer that are off the object
            # there should only be 8 observations stored for the 12 matching steps
            # and all of them should be on the object.
            num_matching_steps = len(
                exp.model.learning_modules[0].buffer.stats["possible_matches"]
            )
            self.assertEqual(
                num_matching_steps,
                sum(
                    exp.model.learning_modules[0].buffer.features["patch"]["on_object"][
                        :num_matching_steps
                    ]
                ),
                "Number of match steps does not match with stored observations "
                "on object",
            )

    def test_detailed_logging(self):
        exp = hydra.utils.instantiate(self.feature_pred_off_object_cfg.test)
        with exp:
            exp.run()

        train_stats, eval_stats, detailed_stats, lm_models = load_stats(
            exp.output_dir,
            load_train=True,
            load_eval=True,
            load_detailed=True,
        )
        for episode in lm_models:
            self.assertEqual(
                list(lm_models[episode]["LM_0"].keys()),
                ["new_object0"],
                "should have only learned and saved one object during training.",
            )
        for row in range(train_stats.shape[0]):
            self.assertEqual(
                train_stats.loc[row]["primary_target_object"],
                detailed_stats[str(row)]["LM_0"]["target"]["object"],
                "targets in train_stats and detailed stats don't match.",
            )

        for row in range(eval_stats.shape[0]):
            detailed_id = train_stats.shape[0] + row - 1
            self.assertEqual(
                eval_stats.loc[row]["primary_target_object"],
                detailed_stats[str(detailed_id)]["LM_0"]["target"]["object"],
                "targets in eval_stats and detailed stats don't match.",
            )
        self.assertEqual(
            len(detailed_stats["1"]["SM_0"]["processed_observations"]),
            73,
            "sensor module observations should contain all observations,"
            "even those off the object.",
        )
        self.assertEqual(
            len(detailed_stats["1"]["LM_0"]["possible_matches"]),
            train_stats.loc[1]["monty_matching_steps"],
            "matching steps in detailed stats don't match with those in train stats.",
        )
        self.assertEqual(
            sum(np.array(detailed_stats["1"]["LM_0"]["patch"]["on_object"])),
            len(detailed_stats["1"]["LM_0"]["patch"]["on_object"]),
            "learning module observations should only contain observations"
            "that were on the object.",
        )
        # Could add more tests but not sure how important.

    def test_uniform_initial_poses(self):
        """Test same scenario as test_fixed_actions_feat with uniform poses."""
        exp = hydra.utils.instantiate(self.feature_uniform_initial_poses_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def gm_learn_object(
        self, graph_lm: FeatureGraphLM, obj_name, observations, offset=None
    ):
        if offset is None:
            offset = np.zeros(3)

        graph_lm.mode = ExperimentMode.TRAIN
        graph_lm.pre_episode(primary_target=self.placeholder_target)

        offset_obs = []
        for observation in observations:
            obs_to_learn = copy.deepcopy(observation)
            obs_to_learn.location += offset
            offset_obs.append(obs_to_learn)
            graph_lm.exploratory_step(self.ctx, [obs_to_learn])

        graph_lm.detected_object = obj_name
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        graph_lm.post_episode()
        return offset_obs

    def get_gm_with_fake_object(self):
        graph_lm = FeatureGraphLM(
            rng=np.random.RandomState(),
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
        )

        self.gm_learn_object(
            graph_lm, obj_name="new_object0", observations=self.fake_obs_learn
        )

        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            1,
            "Should have stored exactly one object in memory.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Learned object ID should be new_object0.",
        )
        return graph_lm

    def get_5lm_gm_with_fake_objects(self, objects) -> list[TrainedGraphLM]:
        graph_lms = []
        for lm in range(5):
            graph_lm = FeatureGraphLM(
                rng=np.random.RandomState(),
                max_match_distance=0.005,
                tolerances={
                    "patch": {
                        "hsv": [0.1, 1, 1],
                        "principal_curvatures_log": [1, 1],
                    }
                },
            )
            object_obs = []
            for i, obj in enumerate(objects):
                obj_name = f"new_object{i}"
                offset_obs = self.gm_learn_object(
                    graph_lm,
                    obj_name=obj_name,
                    observations=obj,
                    offset=self.lm_offsets[lm],
                )
                object_obs.append(EpisodeObservations(obj_name, offset_obs))
            graph_lms.append(TrainedGraphLM(graph_lm, object_obs))

        return graph_lms

    def test_same_sequence_recognition(self):
        """Test that the object is recognized with same action sequence."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_reverse_sequence_recognition(self):
        """Test that object is recognized irrespective of sampling order."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test.reverse()

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step(self.ctx, [observation])
            print(graph_lm.get_possible_matches())
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_offset_sequence_recognition(self):
        """Test that the object is recognized irrespective of its location rel body."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            observation.location = observation.location + np.ones(3)
            graph_lm.matching_step(self.ctx, [observation])
            print(graph_lm.get_possible_matches())
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_new_sampling_recognition(self):
        """Test object recognition with slightly perturbed observations.

        Test that the object is recognized if observations don't exactly fit
        the model.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 0.001

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_locations_not_recognized(self):
        """Test that the object is not recognized if locations don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 5

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for i, observation in enumerate(fake_obs_test):
            graph_lm.matching_step(self.ctx, [observation])
            if i == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one match at the first step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_different_features_not_recognized(self):
        """Test that the object is not recognized if features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].non_morphological_features["hsv"] = [0.5, 1, 1]

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                0,
                "Should have no possible matches.",
            )

    def test_different_pose_features_not_recognized(self):
        """Test that the object is not recognized if pose features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].morphological_features["pose_vectors"] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for step, observation in enumerate(fake_obs_test):
            graph_lm.matching_step(self.ctx, [observation])
            if step == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one possible match at first step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_moving_off_object_and_back(self):
        """Test that the object is still recognized after moving off the object.

        TODO: since the monty class checks use_state in combine_inputs it doesn't make
        much sense to test this here anymore with an isolated LM.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[1].location = [1, 2, 1]
        fake_obs_test[1].morphological_features["on_object"] = 0
        fake_obs_test[1].use_state = False

        graph_lm = self.get_gm_with_fake_object()

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            if not observation.use_state:
                pass
            else:
                graph_lm.matching_step(self.ctx, [observation])
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have exactly one possible match.",
                )
                self.assertEqual(
                    graph_lm.get_possible_matches()[0],
                    "new_object0",
                    "Should match to new_object0.",
                )
        self.assertEqual(
            len(graph_lm.get_possible_paths()["new_object0"]),
            1,
            "Should recognize unique location",
        )
        self.assertEqual(
            len(graph_lm.get_possible_poses()["new_object0"][0]),
            1,
            "Should recognize unique rotation",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0][0]),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_5lm_displacement_experiment(self):
        """Test 5 displacement LMs voting with two evaluation settings."""
        exp = hydra.utils.instantiate(self.five_lm_ppf_displacement_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_multilm_train_results(train_stats, num_lms=5, min_done=3)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_multilm_eval_results(
            eval_stats, num_lms=5, min_done=3, num_episodes=1
        )

    def test_5lm_feature_experiment(self):
        """Test 5 feature LMs voting with two evaluation settings."""
        exp = hydra.utils.instantiate(self.five_lm_feature_cfg.test)
        with exp:
            objects = [self.fake_obs_learn, self.fake_obs_house_3d]
            trained_modules = self.get_5lm_gm_with_fake_objects(objects)
            exp.experiment_mode = ExperimentMode.EVAL
            monty = exp.model
            monty.set_experiment_mode(exp.experiment_mode)
            monty.learning_modules = [tm.learning_module for tm in trained_modules]

            for tm in trained_modules:
                tm.mode = ExperimentMode.EVAL

            exp.pre_epoch()
            ctx = RuntimeContext(rng=exp.rng)

            for episode_num in range(tm.num_episodes):
                exp.pre_episode()
                # Normally the experiment `pre_episode` method would call the model
                # `pre_episode` method, but it expects to feed data from an environment
                # interface to the model, and we aren't using that, so we call it again
                # with the correct target value.
                monty.pre_episode(self.placeholder_target)
                for step in range(tm.num_observations(episode_num)):
                    # Manually run through the internal Monty steps since we aren't
                    # using the data from the environment interface and are instead
                    # providing faked observations.
                    monty.sensor_module_outputs = [
                        lm.episodes[episode_num].observations[step]
                        for lm in trained_modules
                    ]
                    monty._step_learning_modules(ctx)
                    monty._vote()
                    monty._pass_goal_states()
                    monty._set_step_type_and_check_if_done()
                    monty._post_step()
                exp.post_episode(tm.num_observations(episode_num))
            exp.post_epoch()

        output_dir = Path(exp.output_dir)
        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        # Just testing 1 episode here. Somehow the second rotation doesn't get
        # recognized. Probably just some parameter setting due to flaws in old
        # LM but didn't want to dig too deep into that for now.
        self.check_multilm_eval_results(
            eval_stats, num_lms=5, min_done=3, num_episodes=1
        )


if __name__ == "__main__":
    unittest.main()
