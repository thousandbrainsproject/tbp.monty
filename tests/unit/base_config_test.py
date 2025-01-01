# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging
import os
import shutil
import tempfile
import unittest
from pprint import pprint

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    SingleCameraMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    DebugExperimentArgs,
    EnvInitArgsSinglePTZ,
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    NotYCBEvalObjectList,
    NotYCBTrainObjectList,
    SinglePTZHabitatDatasetArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyExperiment


class BaseConfigTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()

        base = dict(
            experiment_class=MontyExperiment,
            experiment_args=DebugExperimentArgs(),
            logging_config=LoggingConfig(
                output_dir=self.output_dir, python_log_level="DEBUG"
            ),
            monty_config=SingleCameraMontyConfig(),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=SinglePTZHabitatDatasetArgs(
                env_init_args=EnvInitArgsSinglePTZ(data_path=None).__dict__
            ),
            train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=NotYCBTrainObjectList().objects,
            ),
            eval_dataloader_class=ED.EnvironmentDataLoaderPerObject,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=NotYCBEvalObjectList().objects,
            ),
        )

        self.base_config = base

        pprint("\n\nCONFIG:\n\n")
        for key, val in self.base_config.items():
            pprint(f"{key}: {val}")

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self):
        """Canary for setup_experiment.

        This could be part of the setUp method, but it's easier to debug if
        something breaks the setup_experiment method if there's a separate test for it.
        """
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)
        self.exp.dataset.close()

    # @unittest.skip("debugging")
    def test_can_run_episode(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)
        pprint("...training...")
        self.exp.model.set_experiment_mode("train")
        self.exp.dataloader = self.exp.train_dataloader
        self.exp.run_episode()
        self.exp.dataset.close()

    # @unittest.skip("speed")
    def test_can_run_train_epoch(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)
        self.exp.model.set_experiment_mode("train")
        self.exp.run_epoch()
        self.exp.dataset.close()

    # @unittest.skip("debugging")
    def test_can_run_eval_epoch(self):
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)
        self.exp.model.set_experiment_mode("eval")
        self.exp.run_epoch()
        self.exp.dataset.close()

    # @unittest.skip("debugging")
    def test_observation_unpacking(self):
        """Make sure this test uses very small n_actions_per_epoch for speed."""
        pprint("...parsing experiment...")
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)

        monty_module_sids = set(
            [s.sensor_module_id for s in self.exp.model.sensor_modules]
        )

        # Handle the training loop manually for this interim test
        max_count = 5
        count = 0
        for observation in self.exp.train_dataloader:
            agent_keys = set(observation.keys())
            sensor_keys = []
            for agent in agent_keys:
                sensor_keys.extend(list(observation[agent].keys()))

            sensor_key_set = set(sensor_keys)
            self.assertCountEqual(
                sensor_key_set, monty_module_sids, "sensor module ids must match"
            )

            count += 1
            if count >= max_count:
                break

        # Verify we can skip the loop and just run a single
        for s in self.exp.model.sensor_modules:
            s_obs = self.exp.model.get_observations(observation, s.sensor_module_id)
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

        self.exp.dataset.close()

    # @unittest.skip("debugging")
    def test_can_save_and_load(self):
        # Make sure deepcopy works (config is serializable)
        config_1 = copy.deepcopy(self.base_config)
        self.assertDictEqual(config_1, self.base_config)

        pprint("...parsing experiment in save and load test...")
        self.exp = MontyExperiment()
        self.exp.setup_experiment(config_1)

        # change something about exp.state that will be saved via state_dict
        new_attr = False
        self.exp.model.learning_modules[0].test_attr_2 = new_attr

        self.exp.save_state_dict()
        prev_model = self.exp.model
        self.exp.dataset.close()

        pprint(f"\n\n\n loading second experiment\n\n\n")
        # checkpoint_dir = self.exp.experiment_args.output_dir
        config_2 = copy.deepcopy(self.base_config)
        config_2["experiment_args"].model_name_or_path = self.exp.output_dir
        exp_2 = MontyExperiment()
        exp_2.setup_experiment(config_2)

        # Test 1: untouched attributes are saved and loaded correctly
        prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
        new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
        self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

        # Test 2: the attribute we changed was saved as such
        new_lm = exp_2.model.learning_modules[0]
        self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

        # Use explicit load_state_dict function instead of setup_experiment
        exp_2.load_state_dict(self.exp.output_dir)

        # Test 1: untouched attributes are saved and loaded correctly
        prev_attr_1_value = prev_model.learning_modules[0].test_attr_1
        new_attr_1_value = exp_2.model.learning_modules[0].test_attr_1
        self.assertEqual(prev_attr_1_value, new_attr_1_value, "attrs do not match")

        # Test 2: the attribute we changed was saved as such
        new_lm = exp_2.model.learning_modules[0]
        self.assertEqual(new_lm.test_attr_2, new_attr, "attrs did not match")

        exp_2.dataset.close()

    def test_logging_debug_level(self):
        """Check that logs go to a file, we can load them, and they have basic info."""
        base_config = copy.deepcopy(self.base_config)
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)

        # Add some stuff to the logs, verify it shows up
        info_message = "INFO is in the log"
        warning_message = "WARNING is in the log"
        logging.info(info_message)
        logging.warning(warning_message)

        with open(os.path.join(self.exp.output_dir, "log.txt"), "r") as f:
            log = f.read()

        self.assertTrue(info_message in log)
        self.assertTrue(warning_message in log)

    def test_logging_info_level(self):
        """Check that if we set logging level to info, debug logs do not show up."""
        base_config = copy.deepcopy(self.base_config)
        base_config["logging_config"].python_log_level = logging.INFO
        self.exp = MontyExperiment()
        self.exp.setup_experiment(base_config)

        # Add some stuff to the logs, verify it shows up
        debug_message = "DEBUG is in the log"
        warning_message = "WARNING is in the log"
        logging.debug(debug_message)
        logging.warning(warning_message)

        with open(os.path.join(self.exp.output_dir, "log.txt"), "r") as f:
            log = f.read()

        self.assertTrue(debug_message not in log)
        self.assertTrue(warning_message in log)


if __name__ == "__main__":
    unittest.main()
