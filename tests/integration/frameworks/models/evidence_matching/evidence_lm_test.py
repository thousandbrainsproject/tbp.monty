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

from tests import HYDRA_ROOT

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import shutil
import tempfile
from pathlib import Path
from pprint import pprint

import hydra
import pandas as pd
from omegaconf import DictConfig

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests.unit.resources.unit_test_utils import BaseGraphTest


class EvidenceLMTest(BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        def hydra_config(test_name: str) -> DictConfig:
            """Return a Hydra configuration from the specified test name.

            Args:
                test_name: the name of the test config to load
            """
            overrides = [
                f"test=evidence_lm/{test_name}",
                f"test.config.logging.output_dir={self.output_dir}",
            ]

            return hydra.compose(config_name="test", overrides=overrides)

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.evidence_cfg = hydra_config("evidence")
            self.fixed_actions_evidence_cfg = hydra_config("fixed_actions_evidence")
            self.evidence_off_object_cfg = hydra_config("evidence_off_object")
            self.evidence_times_out_cfg = hydra_config("evidence_times_out")
            self.uniform_initial_poses_cfg = hydra_config("uniform_initial_poses")
            self.no_features_cfg = hydra_config("no_features")
            self.fixed_possible_poses_cfg = hydra_config("fixed_possible_poses")
            self.five_lm_cfg = hydra_config("five_lm")
            self.five_lm_basic_logging_cfg = hydra_config("five_lm_basic_logging")
            self.five_lm_three_done_cfg = hydra_config("five_lm_three_done")
            self.five_lm_off_object_cfg = hydra_config("five_lm_off_object")
            self.five_lm_no_threading_cfg = hydra_config("five_lm_no_threading")
            self.five_lm_maxnn1 = hydra_config("five_lm_maxnn1")
            self.five_lm_bounded = hydra_config("five_lm_bounded")
            self.noise_mixin_cfg = hydra_config("noise_mixin")
            self.noisy_sensor_cfg = hydra_config("noisy_sensor")

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

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
            exp.model.set_experiment_mode(exp.experiment_mode)
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
            exp.model.set_experiment_mode(exp.experiment_mode)
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
            exp.model.set_experiment_mode(exp.experiment_mode)
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
