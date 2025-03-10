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
import shutil
import tempfile
import unittest
from pprint import pprint

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    PatchAndViewFeatureChangeConfig,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectEvalArgs,
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatDistantPatchSM,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)


class SensorModuleTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()
        base = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20)
            ),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.tested_features = [
            "on_object",
            "object_coverage",
            "rgba",
            "hsv",
            "pose_vectors",
            "principal_curvatures",
            "principal_curvatures_log",
            "gaussian_curvature",
            "mean_curvature",
            "gaussian_curvature_sc",
            "mean_curvature_sc",
        ]

        sensor_feature_test = copy.deepcopy(base)
        sensor_feature_test.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=2),
                sensor_module_configs=dict(
                    sensor_module_0=dict(
                        sensor_module_class=HabitatDistantPatchSM,
                        sensor_module_args=dict(
                            sensor_module_id="patch",
                            features=self.tested_features,
                            save_raw_obs=False,
                        ),
                    ),
                    sensor_module_1=dict(
                        sensor_module_class=DetailedLoggingSM,
                        sensor_module_args=dict(
                            sensor_module_id="view_finder",
                            save_raw_obs=False,
                        ),
                    ),
                ),
            ),
        )

        feature_change_sensor_config = copy.deepcopy(base)
        feature_change_sensor_config.update(
            experiment_args=ExperimentArgs(n_train_epochs=1, n_eval_epochs=1),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewFeatureChangeConfig(
                monty_args=MontyArgs(num_exploratory_steps=100)
            ),
        )

        self.base = base
        self.sensor_feature_test = sensor_feature_test
        self.feature_change_sensor_config = feature_change_sensor_config

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    # @unittest.skip("debugging")
    def test_can_set_up(self):
        """Check that correct features are returned by sensor module."""
        print("...parsing experiment...")
        base_config = copy.deepcopy(self.sensor_feature_test)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.pre_episode()
            for step, observation in enumerate(self.exp.dataloader):
                self.exp.model.step(observation)
                if step == 1:
                    break

    # @unittest.skip("debugging")
    def test_features_in_sensor(self):
        """Check that correct features are returned by sensor module."""
        print("...parsing experiment...")
        base_config = copy.deepcopy(self.sensor_feature_test)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(base_config)
            self.exp.model.set_experiment_mode("train")
            pprint("...training...")
            self.exp.pre_epoch()
            self.exp.pre_episode()
            for _, observation in enumerate(self.exp.dataloader):
                self.exp.model.aggregate_sensory_inputs(observation)

                pprint(self.exp.model.sensor_module_outputs)
                for feature in self.tested_features:
                    if feature in ["pose_vectors", "pose_fully_defined", "on_object"]:
                        self.assertIn(
                            feature,
                            self.exp.model.sensor_module_outputs[
                                0
                            ].morphological_features.keys(),
                            f"{feature} not returned by SM",
                        )
                    else:
                        self.assertIn(
                            feature,
                            self.exp.model.sensor_module_outputs[
                                0
                            ].non_morphological_features.keys(),
                            f"{feature} not returned by SM",
                        )
                break

    def test_feature_change_sm(self):
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.feature_change_sensor_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("...training...")
            self.exp.train()
            # TODO: test that only new features are given to LM
            pprint("...evaluating...")
            self.exp.evaluate()


if __name__ == "__main__":
    unittest.main()
