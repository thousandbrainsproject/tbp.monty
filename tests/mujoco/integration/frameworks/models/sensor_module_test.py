# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import shutil
import tempfile
import unittest

import hydra

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.hydra import instantiate_experiment
from tests import HYDRA_ROOT


class SensorModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
            self.base_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/sensor_module/base_mujoco",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.sensor_feature_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/sensor_module/sensor_feature_mujoco",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.feature_change_sensor_cfg = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=test/sensor_module/feature_change_sensor_mujoco",
                    f"experiment.config.logging.output_dir={self.output_dir}",
                ],
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_can_set_up(self) -> None:
        exp = instantiate_experiment(self.base_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.pre_episode()
            ctx = RuntimeContext(rng=exp.rng)
            actions: list[Action] = []
            for _ in range(2):
                observations, proprioceptive_state = exp.env_interface.step(actions)
                actions = exp.model.step(ctx, observations, proprioceptive_state)

    def test_features_in_sensor(self) -> None:
        """Check that correct features are returned by sensor module."""
        exp = instantiate_experiment(self.sensor_feature_cfg.experiment)
        with exp:
            exp.experiment_mode = ExperimentMode.TRAIN
            exp.model.set_experiment_mode(exp.experiment_mode)
            exp.pre_epoch()
            exp.pre_episode()
            ctx = RuntimeContext(rng=exp.rng)
            observations, proprioceptive_state = exp.env_interface.step([])
            exp.model.aggregate_sensory_inputs(ctx, observations, proprioceptive_state)

            # Dig the features list out of the hydra config
            config = self.sensor_feature_cfg.experiment.config
            sensor_modules = config.monty_config.sensor_modules
            tested_features = sensor_modules.sensor_module_0.features
            for feature in tested_features:
                if feature in ["pose_vectors", "pose_fully_defined", "on_object"]:
                    self.assertIn(
                        feature,
                        exp.model.sensor_module_outputs[
                            0
                        ].morphological_features.keys(),
                        f"{feature} not returned by SM",
                    )
                else:
                    self.assertIn(
                        feature,
                        exp.model.sensor_module_outputs[
                            0
                        ].non_morphological_features.keys(),
                        f"{feature} not returned by SM",
                    )

    def test_feature_change_sm(self):
        exp = instantiate_experiment(self.feature_change_sensor_cfg.experiment)
        with exp:
            exp.run()
            # TODO: test that only new features are given to LM
