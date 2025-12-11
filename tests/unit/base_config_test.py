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

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import logging
import shutil
import tempfile
import unittest
from typing import Mapping

import hydra
from omegaconf import OmegaConf


class BaseConfigTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.base_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=base_config/base",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self):
        """Canary for setup_experiment.

        This could be part of the setUp method, but it's easier to debug if
        something breaks the setup_experiment method if there's a separate test for it.
        """
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            pass

    # @unittest.skip("debugging")
    def test_can_run_episode(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.model.set_experiment_mode("train")
            exp.env_interface = exp.train_env_interface
            exp.run_episode()

    # @unittest.skip("speed")
    def test_can_run_train_epoch(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.model.set_experiment_mode("train")
            exp.run_epoch()

    # @unittest.skip("debugging")
    def test_can_run_eval_epoch(self):
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            exp.model.set_experiment_mode("eval")
            exp.run_epoch()

    # @unittest.skip("debugging")
    def test_observation_unpacking(self):
        """Make sure this test uses very small n_actions_per_epoch for speed."""
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            monty_module_sids = {s.sensor_module_id for s in exp.model.sensor_modules}

            # Handle the training loop manually for this interim test
            max_count = 5
            for count, observation in enumerate(exp.train_env_interface):
                agent_keys = set(observation.keys())
                sensor_keys = []
                for agent in agent_keys:
                    sensor_keys.extend(list(observation[agent].keys()))

                sensor_key_set = set(sensor_keys)
                self.assertCountEqual(
                    sensor_key_set, monty_module_sids, "sensor module ids must match"
                )

                if count >= max_count:
                    break

            # Verify we can skip the loop and just run a single
            for s in exp.model.sensor_modules:
                s_obs = exp.model.get_observations(observation, s.sensor_module_id)
                feature = s.step(s_obs)
                self.assertIn(
                    "rgba",
                    feature.non_morphological_features,
                    "sensor_module must receive rgba",
                )
                self.assertIn(
                    "depth",
                    feature.non_morphological_features,
                    "sensor_module must receive depth",
                )

    # @unittest.skip("debugging")
    def test_can_save_and_load(self):
        # Make sure deepcopy works (config is serializable)
        config_1 = OmegaConf.to_object(self.base_cfg)

        exp = hydra.utils.instantiate(config_1["test"])
        with exp:
            # change something about exp.state that will be saved via state_dict
            new_attr = False
            exp.model.learning_modules[0].test_attr_2 = new_attr

            exp.save_state_dict()
            prev_model = exp.model

        config_2: Mapping = OmegaConf.to_object(  # ignore: type[assignment]
            self.base_cfg
        )
        config_2["test"]["config"]["model_name_or_path"] = exp.output_dir

        exp_2 = hydra.utils.instantiate(config_2["test"])
        with exp_2:
            # Test 1: untouched attributes are saved and loaded correctly
            prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
            new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
            self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

            # Test 2: the attribute we changed was saved as such
            new_lm = exp_2.model.learning_modules[0]
            self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

            # Use explicit load_state_dict function instead of setup_experiment
            exp_2.load_state_dict(exp.output_dir)

            # Test 1: untouched attributes are saved and loaded correctly
            prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
            new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
            self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

            # Test 2: the attribute we changed was saved as such
            new_lm = exp_2.model.learning_modules[0]
            self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

    def test_logging_debug_level(self) -> None:
        """Check that logs go to a file, we can load them, and they have basic info."""
        exp = hydra.utils.instantiate(self.base_cfg.test)
        with exp:
            # Add some stuff to the logs, verify it shows up
            info_message = "INFO is in the log"
            warning_message = "WARNING is in the log"

            logger = logging.getLogger("tbp.monty")
            logger.info(info_message)
            logger.warning(warning_message)

            with (exp.output_dir / "log.txt").open() as f:
                log = f.read()

            self.assertTrue(info_message in log)
            self.assertTrue(warning_message in log)

    def test_logging_info_level(self) -> None:
        """Check that if we set logging level to info, debug logs do not show up."""
        base_config: Mapping = OmegaConf.to_object(self.base_cfg)
        base_config["test"]["config"]["logging"]["python_log_level"] = logging.INFO

        exp = hydra.utils.instantiate(base_config["test"])
        with exp:
            # Add some stuff to the logs, verify it shows up
            debug_message = "DEBUG is in the log"
            warning_message = "WARNING is in the log"

            logger = logging.getLogger("tbp.monty")
            logger.debug(debug_message)
            logger.warning(warning_message)

            with (exp.output_dir / "log.txt").open() as f:
                log = f.read()

            self.assertTrue(debug_message not in log)
            self.assertTrue(warning_message in log)


if __name__ == "__main__":
    unittest.main()
