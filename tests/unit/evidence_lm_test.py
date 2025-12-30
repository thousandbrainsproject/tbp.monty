# Copyright 2025 Thousand Brains Project
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

from tbp.monty.frameworks.experiments.monty_experiment import ExperimentMode

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tests.unit.resources.unit_test_utils import BaseGraphTest


class EvidenceLMTest(BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.default_gsg_config = dict(
            elapsed_steps_factor=10,
            min_post_goal_success_steps=5,
            x_percent_scale_factor=0.75,
            desired_object_distance=0.03,
        )
        self.output_dir = tempfile.mkdtemp()
        self.fixed_actions_path = (
            Path(__file__).parent / "resources" / "fixed_test_actions.jsonl"
        )
        self.fixed_actions_off_object_path = (
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
                "policy_args",
                "file_name",
            ]
        )

        def hydra_config(
            test_name: str, action_file_name: Path | None = None
        ) -> DictConfig:
            """Return a Hydra configuration from the specified test name.

            Args:
                test_name: the name of the test config to load
                action_file_name: Optional path to a file of actions to use
            """
            overrides = [
                f"test=evidence_lm/{test_name}",
                f"test.config.logging.output_dir={self.output_dir}",
            ]
            if action_file_name:
                overrides.append(f"{actions_file_name_selector}={action_file_name}")

            return hydra.compose(config_name="test", overrides=overrides)

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.evidence_cfg = hydra_config("evidence")
            self.fixed_actions_evidence_cfg = hydra_config(
                "fixed_actions_evidence", self.fixed_actions_path
            )
            self.evidence_off_object_cfg = hydra_config(
                "evidence_off_object", self.fixed_actions_off_object_path
            )
            self.evidence_times_out_cfg = hydra_config(
                "evidence_times_out", self.fixed_actions_path
            )
            self.uniform_initial_poses_cfg = hydra_config(
                "uniform_initial_poses", self.fixed_actions_path
            )
            self.no_features_cfg = hydra_config("no_features", self.fixed_actions_path)
            self.fixed_possible_poses_cfg = hydra_config(
                "no_features", self.fixed_actions_path
            )
            self.five_lm_cfg = hydra_config("five_lm", self.fixed_actions_path)
            self.five_lm_basic_logging_cfg = hydra_config(
                "five_lm_basic_logging", self.fixed_actions_path
            )
            self.five_lm_three_done_cfg = hydra_config(
                "five_lm_three_done", self.fixed_actions_path
            )
            self.five_lm_off_object_cfg = hydra_config(
                "five_lm_off_object", self.fixed_actions_off_object_path
            )
            self.five_lm_no_threading_cfg = hydra_config(
                "five_lm_no_threading", self.fixed_actions_path
            )
            self.five_lm_maxnn1 = hydra_config(
                "five_lm_maxnn1", self.fixed_actions_path
            )
            self.five_lm_bounded = hydra_config(
                "five_lm_bounded", self.fixed_actions_path
            )
            self.noise_mixin_cfg = hydra_config("noise_mixin", self.fixed_actions_path)
            self.noisy_sensor_cfg = hydra_config(
                "noisy_sensor", self.fixed_actions_path
            )

        """
        TESTS STILL MISSING:
        Most of them should probably be added in a separate file. We could have a test
        suite for utils etc. and one for policies. For now I grouped them by file.

        general:
        - [!] Surface-agent sensor + all its policies
        - Surface-agent sensor online plotting (show_sensor_output)

        monty_custom_coponents:
        - [!] Test feature change SM

        Graph utils
        - [!] Build graph from repeated observations and check that similar ones are
          removed (also if location is similar but surface normal is not)
        - Test check_orthonormal function
        - [!] Test get center surface normal under different off object conditions
        - [!] already_in_list with using all features for graph building

        transforms:
        - get_semantic_from_depth under different conditions

        embodied_data:
        - move_close_enough
        - find object if not in view

        object_recognition_experiments:
        - MontyGeneralizationExperiment

        graph_matching_loggers:
        - SelectiveEvidenceLogger

        Could also be nice to sort these tests some time.
        """

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def get_elm_with_fake_object(
        self,
        fake_obs,
        initial_possible_poses="informed",
        gsg_class=None,
        gsg_args=None,
    ):
        graph_lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            # set graph size larger since fake obs displacements are meters
            max_graph_size=10,
            gsg_class=gsg_class,
            gsg_args=gsg_args,
            hypotheses_updater_args=dict(
                initial_possible_poses=initial_possible_poses,
            ),
        )
        graph_lm.mode = "train"
        for observation in fake_obs:
            graph_lm.exploratory_step([observation])
        graph_lm.detected_object = "new_object0"
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs),
            f"Should have stored exactly {fake_obs} locations in the buffer.",
        )
        graph_lm.post_episode()
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

    def get_elm_with_two_fake_objects(
        self, fake_obs, fake_obs_two, initial_possible_poses, gsg_class, gsg_args
    ) -> EvidenceGraphLM:
        """Train on two fake observation objects.

        Returns:
            Evidence GraphLearning Module
        """
        # Train on first object
        graph_lm = self.get_elm_with_fake_object(
            fake_obs,
            initial_possible_poses=initial_possible_poses,
            gsg_class=gsg_class,
            gsg_args=gsg_args,
        )

        # Train on second object
        obj_two_target = copy.deepcopy(self.placeholder_target)
        obj_two_target["object"] = "new_object1"
        graph_lm.pre_episode(primary_target=obj_two_target)
        for observation in fake_obs_two:
            graph_lm.exploratory_step([observation])
        graph_lm.detected_object = obj_two_target["object"]
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs_two),
            f"Should have stored exactly {fake_obs_two} locations in the buffer.",
        )
        graph_lm.post_episode()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            2,
            "Should have stored two objects in memory now.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Should still know first object.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[1],
            "new_object1",
            "Learned object ID should be new_object1.",
        )

        return graph_lm

    def test_can_run_evidence_experiment(self):
        exp = hydra.utils.instantiate(self.evidence_cfg.test)
        with exp:
            exp.run()

    def test_fixed_actions_evidence(self):
        """Test 3 train and 3 eval epochs with 2 objects and 2 rotations."""
        exp = hydra.utils.instantiate(self.fixed_actions_evidence_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")

        self.check_eval_results(eval_stats)
        for key in [
            "possible_rotations",
            "possible_locations",
            "evidences",
            "symmetry_evidence",
        ]:
            self.assertIn(
                key,
                exp.model.learning_modules[0].buffer.stats.keys(),
                f"{key} should be stored in buffer when using DETAILED logging.",
            )
        self.assertGreater(
            len(exp.model.learning_modules[0].buffer.stats["possible_matches"]),
            1,
            "When using detailed logging we should store matches at every steps.",
        )

    def test_pre_episode_raises_error_when_no_object_is_present(self):
        """Test that pre_episode raises an error when no object is present."""
        exp = hydra.utils.instantiate(self.fixed_actions_evidence_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.env._env.remove_all_objects()
            with self.assertRaises(ValueError) as error:
                exp.pre_episode()
            self.assertEqual(
                "May be initializing experiment with no visible target object",
                str(error.exception),
            )

    def test_moving_off_object(self):
        """Test logging when moving off the object for some steps during an episode."""
        exp = hydra.utils.instantiate(self.evidence_off_object_cfg.test)
        with exp:
            # First episode will be used to learn object (no_match is triggered before
            # min_steps is reached and the sensor moves off the object). In the second
            # episode the sensor moves off the sphere on episode steps 6+

            # The off_object points are not counted as steps. Therefore we have to wait
            # until the camera turns a full circle and arrives on the other side of the
            # object. From there we can continue to try and recognize the object.

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
            "Number of match steps does not match with stored observations on object",
        )
        # Since min_train_steps==12 we should have taken 13 steps.
        self.assertEqual(
            exp.model.matching_steps,
            13,
            "Did not take correct amount of matching steps. Perhaps "
            "min_train_steps was not applied correctly.",
        )
        self.assertGreater(
            exp.model.episode_steps,
            exp.model.matching_steps + 80,
            "number of episode steps should be larger than matching steps + "
            "exploration steps since we don't count off object observations as"
            "as matching steps (but do count them as episode steps).",
        )

        self.assertGreater(
            exp.model.learning_modules[0].buffer.stats["current_mlh"][-1]["evidence"],
            exp.model.learning_modules[0].buffer.stats["current_mlh"][6]["evidence"],
            "evidence should have increased after moving back on the object.",
        )

    def test_evidence_time_out(self):
        exp = hydra.utils.instantiate(self.evidence_times_out_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.assertEqual(
            train_stats["individual_ts_performance"][0],
            "no_match",
            "with no objects in memory individual_ts_performance should be no match",
        )
        for i in range(5):
            self.assertEqual(
                train_stats["individual_ts_performance"][i + 1],
                "time_out",
                f"time out not recognized/logged correctly in episode {i + 1}",
            )
        self.assertEqual(
            train_stats["primary_performance"][2],
            "correct_mlh",
            "Evidence LM should look at most likely hypothesis at time out",
        )
        self.assertEqual(
            train_stats["primary_performance"][3],
            "confused_mlh",
            "unknown object should be logged as confused_mlh at time out",
        )
        self.assertEqual(
            len(exp.model.learning_modules[0].get_all_known_object_ids()),
            1,
            "No new objects should be added to memory after time out.",
        )
        self.assertLessEqual(
            exp.model.learning_modules[0]
            .get_graph("new_object0", input_channel="first")
            .x.shape[1],
            32,  # max_train_steps + exploratory_steps
            "No new points should be added to an existing graph after time out.",
        )

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        for i in range(3):
            self.assertEqual(
                eval_stats["individual_ts_performance"][i],
                "time_out",
                f"time out not recognized/logged correctly in eval episode {i}",
            )
            self.assertEqual(
                eval_stats["primary_performance"][i],
                "correct_mlh",
                f"time out should use mlh in eval episode {i}",
            )

    def test_evidence_confused_logging(self):
        # When the algorithm evolves, this scenario may not lead to confusion
        # anymore. Setting min_steps would also avoid this, probably.
        exp = hydra.utils.instantiate(self.fixed_actions_evidence_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
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

    def test_uniform_initial_poses(self):
        """Test same scenario as test_fixed_actions_evidence with uniform poses."""
        exp = hydra.utils.instantiate(self.uniform_initial_poses_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")

        self.check_eval_results(eval_stats)

    def test_fixed_initial_poses(self):
        """Test same scenario as test_fixed_actions_evidence with predefined poses."""
        exp = hydra.utils.instantiate(self.fixed_possible_poses_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def test_symmetry_recognition(self):
        """Test that symmetry is recognized."""
        fake_obs_test = copy.deepcopy(self.fake_obs_symmetric)
        # Get LM with object learned from fake_obs
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_symmetric)

        graph_lm.mode = "eval"
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        num_steps_checked_symmetry = 0
        for i in range(12):
            observation = fake_obs_test[i % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
            # since we don't have a monty class here we have to call this check
            # manually. Usually monty class coordinates terminal condition checks and
            # updates to symmetry count.
            graph_lm.get_unique_pose_if_available("new_object0")
            max_obj_evidence = np.max(graph_lm.evidence["new_object0"])
            if max_obj_evidence > graph_lm.object_evidence_threshold:
                num_steps_checked_symmetry += 1
                # On the first step we just store previous hypothesis ids.
                if num_steps_checked_symmetry > 1:
                    # On the second step we will add 1 symmetry evidence
                    # every step because we can't resolve between
                    # 0,0,0 and 180, 0, 180 rotation.
                    self.assertEqual(
                        graph_lm.symmetry_evidence,
                        num_steps_checked_symmetry - 1,
                        "Symmetry evidence doesn't seem to be as expected.",
                    )
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
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0]),
            [0, 0, 0],
            "Should have rotation 0, 0, 0 as a possible pose.",
        )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][-1]),
            [180.0, 0.0, 180.0],
            "Since have symmtry here 180, 0, 180 should also be a possible pose.",
        )

    def test_same_sequence_recognition_elm(self):
        """Test that the object is recognized with same action sequence."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
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
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_reverse_sequence_recognition_elm(self):
        """Test that object is recognized irrespective of sampling order."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test.reverse()

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
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
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_offset_sequence_recognition_elm(self):
        """Test that the object is recognized irrespective of its location rel body."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            observation.location = observation.location + np.ones(3)
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
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
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_new_sampling_recognition_elm(self):
        """Test object recognition with slightly perturbed observations.

        Test that the object is recognized if observations don't exactly fit
        the model.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 0.001
        fake_obs_test[2].location = fake_obs_test[2].location + np.ones(3) * 0.002

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
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
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_locations_not_recognized_elm(self):
        """Test that the object is not recognized if locations don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 5

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for i, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
            if i < 2:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one match at the first and second step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_channel_mapper_shape_elm(self):
        """Test that the channel mapper matches evidence keys and shape."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        graph_lm.matching_step([fake_obs_test[0]])

        self.assertEqual(
            graph_lm.evidence.keys(),
            graph_lm.channel_hypothesis_mapping.keys(),
            "Graph ID should match.",
        )

        self.assertEqual(
            graph_lm.evidence["new_object0"].shape[0],
            graph_lm.channel_hypothesis_mapping["new_object0"].total_size,
            "Channel mapper should have the total number of hypotheses in evidence",
        )

    def _evaluate_target_location(
        self, graph_lm, fake_obs_test, target_object, focus_on_pose=False
    ):
        """Helper function for hypothesis testing that retreives a target location."""
        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)

        # Observe 4 / 5 of the available features
        for ii in range(4):
            observation = fake_obs_test[ii]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])

        if not focus_on_pose:
            # Since up to now we had identical evidence for both cube and house, we give
            # the house an edge now so we test for it and get the expected result.
            graph_lm.current_mlh["graph_id"] = "new_object1"
            graph_lm.evidence["new_object1"] += 1

        # Based on most recent observation, propose the most misaligned graph
        # sub-regions
        graph_lm.gsg.focus_on_pose = focus_on_pose
        target_loc_id = graph_lm.gsg._compute_graph_mismatch()

        target_graph = graph_lm.get_graph(target_object)
        first_input_channel = graph_lm.get_input_channels_in_graph(target_object)[0]
        target_loc = target_graph[first_input_channel].pos[target_loc_id]

        assert np.all(np.isclose(target_loc, self.fake_obs_house[4].location)), (
            "Should propose testing 5th (indexed from 0) location on object"
        )

    def test_hypothesis_testing_proposal_for_id(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        The "square" object is made up of four simple features, while the "house"
        object is a 2D square with an additional, 5th feature above (like the point
        on a house's roof); these two objects are a 2D analogue for distinguishing e.g.
        a mug with a handle from a cylindrical can.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=self.default_gsg_config,
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_id_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        As for test_hypothesis_testing_proposal_for_id, but in this case, the
        "house" object has been rotated and translated in the environment, and we check
        that the policy still proposes the correct location in object-centric
        coordinates.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house_trans)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=self.default_gsg_config,
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_pose(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0], [0, 0, 180]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (rotated
            # about z)
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=self.default_gsg_config,
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_hypothesis_testing_proposal_for_pose_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        In addition, the house has been transformed in environmental coordinates
        (translated and rotated), and we confirm that the object-centric target
        point is still correct.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house_trans)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190], [315, 285, 10]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (was rotated
            # about z before the additional complex transformation was applied)
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=self.default_gsg_config,
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_different_features_still_recognized(self):
        """Test that the object is still recognized if features don't match.

        The evidence LM just uses features as an aid to recognize objects faster
        but should still be able to recognize an object if features don't match
        as long as the morphology matches.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[1].non_morphological_features["hsv"] = [0.9, 1, 1]
        fake_obs_test[2].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[3].non_morphological_features["hsv"] = [0.7, 1, 1]

        for i in range(4):
            fake_obs_test[i].non_morphological_features["principal_curvatures_log"] = [
                10,
                10,
            ]

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        # We start at evidence 0 since we don't get feature evidence at initialization
        for target_evidence, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly 1 possible match.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since none of the features match we should only be getting evidence "
                "from morphology which is 1 at each step (since m matches perfectly).",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_pose_features_not_recognized_elm(self):
        """Test that the object is not recognized if pose features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].morphological_features["pose_vectors"] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for step, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step([observation])
            if step == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one possible match at first step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "First step should have evidence 1 since features match and poses"
                    " are just being initialized.",
                )
            elif step == 1:
                self.assertGreater(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Second step should still have evidence >1 since initialized pose"
                    " is still possible.",
                )
            elif step == 2:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Third step should have evidence <1 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
            elif step == 3:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    0,
                    "Fourth step should have evidence <0 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_moving_off_object_and_back_elm(self):
        """Test that the object is still recognized after moving off the object.

        Test that the object is still recognized after moving off the object
        and that evidence is not incremented in that step.

        TODO: since the monty class checks use_state in combine_inputs it doesn't make
        much sense to test this here anymore with an isolated LM.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[1].location = [1, 2, 1]
        fake_obs_test[1].morphological_features["on_object"] = 0
        fake_obs_test[1].use_state = False

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = "eval"
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for step, observation in enumerate(fake_obs_test):
            if not observation.use_state:
                pass
            else:
                graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
                graph_lm.matching_step([observation])
                # If we are off the object, no evidence should be added
                if observation.morphological_features["on_object"] != 0 and step > 0:
                    target_evidence += 2
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
                    graph_lm.get_current_mlh()["graph_id"],
                    "new_object0",
                    "new_object0 should be the mlh at every step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    target_evidence,
                    "Since we have exact matches, evidence should be incremented by "
                    "2 at each step (1 for pose match and 1 for feature match).",
                )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_can_run_with_no_features(self):
        """Standard evaluation setup but using only pose features."""
        exp = hydra.utils.instantiate(self.no_features_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats)

    def test_5lm_evidence_experiment(self):
        """Test 5 evidence LMs voting with two evaluation settings."""
        exp = hydra.utils.instantiate(self.five_lm_cfg.test)
        with exp:
            exp.train()

            output_dir = Path(exp.output_dir)
            train_stats = pd.read_csv(output_dir / "train_stats.csv")
            print(train_stats)
            self.check_train_results(train_stats, num_lms=5)

            # TODO: Don't manually fake evaluation. Run this experiment
            # as normal and create a follow-up experiment for second evaluation.
            exp.experiment_mode = ExperimentMode.EVAL
            exp.logger_handler.pre_eval(exp.logger_args)
            exp.model.set_experiment_mode("eval")
            for _ in range(exp.n_eval_epochs):
                exp.run_epoch()
            exp.logger_handler.post_eval(exp.logger_args)
            eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
            self.check_eval_results(eval_stats, num_lms=5)

            for lm in exp.model.learning_modules:
                lm.max_match_distance = 0.01
            exp.evaluate()

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")

        self.check_eval_results(eval_stats, num_lms=5)

    def test_5lm_3done_evidence(self):
        """Test 5 evidence LMs voting works with lower min_lms_match setting."""
        exp = hydra.utils.instantiate(self.five_lm_three_done_cfg.test)
        with exp:
            exp.train()

            output_dir = Path(exp.output_dir)
            train_stats = pd.read_csv(output_dir / "train_stats.csv")
            self.check_multilm_train_results(train_stats, num_lms=5, min_done=3)
            # TODO: Don't reach into the internals. Create another experiment for this.
            # Same as in previous test we make it a bit more difficult during eval
            for lm in exp.model.learning_modules:
                lm.max_match_distance = 0.01
            exp.evaluate()

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")

        self.check_multilm_eval_results(eval_stats, num_lms=5, min_done=3)

    def test_moving_off_object_5lms(self):
        """Test logging when moving off the object for some steps during an episode.

        TODO: This test doesn't check if the voting evidence is incremented correctly
          with some LMs off the object. Actually, we still need to decide on some
          protocols for that. Like does the LM still get to vote? Does it still receive
          votes?
        """
        exp = hydra.utils.instantiate(self.five_lm_off_object_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        # Just checking that objects are still recognized correctly when moving off
        # the object.
        self.check_train_results(train_stats, num_lms=5)
        # now lets check the number of steps
        for row in range(5 * 6):
            self.assertLess(
                train_stats["num_steps"][row],
                train_stats["monty_steps"][row],
                "Should have less steps per lm (matching and exploratory"
                " steps that were on the object) than monty_steps.",
            )
            self.assertLess(
                train_stats["monty_matching_steps"][row],
                train_stats["num_steps"][row],
                "Should have less steps monty_matching_steps than"
                " overall steps since some steps were off the object.",
            )

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats, num_lms=5)

        # now lets check the number of steps
        for row in range(5 * 3):
            self.assertLess(
                eval_stats["monty_matching_steps"][row],
                eval_stats["monty_steps"][row],
                "Should have less matching steps than overall monty steps since we"
                " were completely off the object for a while.",
            )
            self.assertGreaterEqual(
                eval_stats["monty_matching_steps"][row],
                eval_stats["num_steps"][row],
                "Individual lms shhould not have more steps that monty matching"
                " steps during evaluation (no exploration).",
            )
            self.assertEqual(
                eval_stats["monty_matching_steps"][row],
                13,
                "All eval episodes should have recognized the object after 13 steps.",
            )
        for lm_id in [0, 2, 3, 4]:
            self.assertLess(
                eval_stats["num_steps"][5 + lm_id],
                13,
                f"LM {lm_id} in episode 1 should have less than 13 steps"
                f" since it was off the object for longer than other LMs.",
            )

    def test_5lms_pre_episode_raises_error_when_no_object_is_present(self):
        """Test that pre_episode raises an error when no object is present."""
        pprint("...parsing experiment...")
        exp = hydra.utils.instantiate(self.five_lm_cfg.test)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode("train")
            exp.pre_epoch()
            exp.env._env.remove_all_objects()
            with self.assertRaises(ValueError) as error:
                exp.pre_episode()
            self.assertEqual(
                "May be initializing experiment with no visible target object",
                str(error.exception),
            )

    def test_5lm_basic_logging(self):
        """Test that 5LM setup works with BASIC logging and stores correct data."""
        exp = hydra.utils.instantiate(self.five_lm_basic_logging_cfg.test)
        with exp:
            exp.run()
            for key in [
                "possible_rotations",
                "possible_locations",
                "evidences",
                "symmetry_evidence",
            ]:
                self.assertNotIn(
                    key,
                    exp.model.learning_modules[0].buffer.stats.keys(),
                    f"{key} should not be stored in buffer when using BASIC logging.",
                )
            self.assertEqual(
                len(exp.model.learning_modules[0].buffer.stats["possible_matches"]),
                1,
                "When using basic logging we don't append stats for every step.",
            )

    def test_can_run_with_no_multithreading_5lms(self):
        """Standard evaluation setup but using only pose features.

        Testing with 5LMs since voting also uses multithreading.
        """
        exp = hydra.utils.instantiate(self.five_lm_no_threading_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats, num_lms=5)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats, num_lms=5)

    def test_can_run_with_maxnn1_5lms(self):
        """Standard evaluation setup but using max_nneighbors=1.

        Testing with 5LMs since voting also uses max_nneighbors.
        """
        exp = hydra.utils.instantiate(self.five_lm_maxnn1.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats, num_lms=5)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats, num_lms=5)

    def test_can_run_with_bounded_evidence_5lms(self):
        """Standard evaluation setup with 5lm and bounded evidence."""
        exp = hydra.utils.instantiate(self.five_lm_bounded.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        self.check_train_results(train_stats, num_lms=5)

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        self.check_eval_results(eval_stats, num_lms=5)

    def test_noise_mixing_evidence(self):
        """Test standard fixed action setting with noisy sensor module.

        NOTE: This test only checks that noise is being applied. It doesnt check
        the models noise robustness. We may want to add a test for that but with
        the current test setup we have too few too similar objects to set parameters
        in a good way.
        """
        exp = hydra.utils.instantiate(self.noise_mixin_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        # NOTE: This might fail if the model becomes more noise robust or
        # better able to deal with few incomplete objects in memory.
        for i in range(6):
            self.assertEqual(
                train_stats["primary_performance"][i],
                "no_match",
                f"Train episode {i} didnt reach no_match."
                "Is noise being applied correctly?",
            )

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        for i in range(3):
            self.assertEqual(
                eval_stats["primary_performance"][i],
                "no_match",
                f"Eval episode {i} didnt reach no_match."
                "Is noise being applied correctly?",
            )

    def test_raw_sensor_noise_evidence(self):
        """Test standard fixed action setting with raw sensor noise.

        NOTE: Same as above, this test only checks that noise is being applied.
        It doesnt check the models noise robustness.

        TODO: Make this test run faster
        """
        exp = hydra.utils.instantiate(self.noisy_sensor_cfg.test)
        with exp:
            exp.run()

        output_dir = Path(exp.output_dir)
        train_stats = pd.read_csv(output_dir / "train_stats.csv")
        # NOTE: This might fail if the model becomes more noise robust or
        # better able to deal with few incomplete objects in memory.
        for i in range(6):
            self.assertEqual(
                train_stats["primary_performance"][i],
                "no_match",
                f"Train episode {i} didnt reach no_match."
                "Is noise being applied correctly?",
            )

        eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
        for i in range(3):
            self.assertEqual(
                eval_stats["primary_performance"][i],
                "no_match",
                f"Eval episode {i} didnt reach no_match."
                "Is noise being applied correctly?",
            )


if __name__ == "__main__":
    unittest.main()
