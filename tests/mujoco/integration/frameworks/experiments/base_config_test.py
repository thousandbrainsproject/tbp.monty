# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import shutil
import tempfile
import unittest
from typing import Mapping

import hydra
from omegaconf import OmegaConf

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tests import HYDRA_ROOT


class BaseConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.base_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/base_config/base_mujoco",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
        with exp:
            pass

    def test_can_run_episode(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.env_interface = exp.train_env_interface
            exp.run_episode()

    def test_can_run_train_epoch(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.run_epoch()

    def test_can_run_eval_epoch(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.EVAL
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.run_epoch()

    def test_observation_unpacking(self) -> None:
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
        with exp:
            monty_module_sids = {s.sensor_module_id for s in exp.model.sensor_modules}

            # Handle the training loop manually for this interim test
            max_count = 5
            count = 0
            ctx = RuntimeContext(rng=exp.rng)
            while True:
                observations, _ = exp.train_env_interface.step([])
                agent_keys = set(observations.keys())
                sensor_keys = []
                for agent in agent_keys:
                    sensor_keys.extend(list(observations[agent].keys()))

                sensor_key_set = set(sensor_keys)
                self.assertCountEqual(
                    sensor_key_set, monty_module_sids, "sensor module ids must match"
                )

                if count >= max_count:
                    break

                count += 1

            # Verify we can skip the loop and just run a single
            for s in exp.model.sensor_modules:
                s_obs = exp.model.get_observations(observations, s.sensor_module_id)
                feature = s.step(ctx, s_obs)
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

    def test_can_save_and_load(self) -> None:
        config_1: Mapping = OmegaConf.to_object(self.base_cfg)  # type: ignore[assignment,type-arg]

        exp = hydra.utils.instantiate(config_1["experiment"])
        with exp:
            # Change something about exp.state that will be saved via save_state_dir.
            new_attr = False
            exp.model.learning_modules[0].test_attr_2 = new_attr

            exp.save_state_dir()
            prev_model = exp.model

        config_2: Mapping = OmegaConf.to_object(  # type: ignore[assignment,type-arg]
            self.base_cfg
        )
        config_2["experiment"]["config"]["model_name_or_path"] = exp.output_dir

        exp_2 = hydra.utils.instantiate(config_2["experiment"])
        with exp_2:
            # Test 1: untouched attributes are saved and loaded correctly
            prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
            new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
            self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

            # Test 2: the attribute we changed was saved as such
            new_lm = exp_2.model.learning_modules[0]
            self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

            # Use explicit load_state_dir function instead of setup_experiment
            exp_2.load_state_dir(exp.output_dir)

            # Test 1: untouched attributes are saved and loaded correctly
            prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
            new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
            self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

            # Test 2: the attribute we changed was saved as such
            new_lm = exp_2.model.learning_modules[0]
            self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

    def test_logging_debug_level(self) -> None:
        """Check that logs go to a file, we can load them, and they have basic info."""
        # TODO: This seems to test the behaviour of the Python `logging` library, which
        #   we should just assume works as advertised. Change this test to introspect
        #   the experiment to see if the loggers got configured correctly instead.
        exp = hydra.utils.instantiate(self.base_cfg.experiment)
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
        # TODO: This seems to test the behaviour of the Python `logging` library, which
        #   we should just assume works as advertised. Change this test to introspect
        #   the experiment to see if the loggers got configured correctly instead.
        base_config: Mapping = OmegaConf.to_object(self.base_cfg)  # type: ignore[assignment,type-arg]
        log_cfg = base_config["experiment"]["config"]["logging"]
        log_cfg["python_log_level"] = logging.INFO

        exp = hydra.utils.instantiate(base_config["experiment"])
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
