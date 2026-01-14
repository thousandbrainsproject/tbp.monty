# Copyright 2026 Thousand Brains Project
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

from unittest import TestCase

import hydra

from tbp.monty.frameworks.run import output_dir_from_run_name


class TutorialsTest(TestCase):
    def test_first_experiment(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/first_experiment"],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

    def test_training_and_inference(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/surf_agent_2obj_train"],
            )
            config.experiment.config.logging.output_dir = str(
                output_dir_from_run_name(config)
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=tutorial/surf_agent_2obj_eval",
                    # We don't need to run the whole thing.
                    "experiment.config.n_eval_epochs=1",
                    "experiment.config.max_eval_steps=3",
                    "experiment.config.max_total_steps=3",
                ],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

    def test_unsupervised_continual_learning(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=tutorial/surf_agent_2obj_unsupervised",
                    # We don't need to run the whole thing.
                    "experiment.config.n_train_epochs=1",
                    "experiment.config.max_train_steps=3",
                    "experiment.config.max_total_steps=3",
                ],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

    def test_multiple_learning_modules(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/dist_agent_5lm_2obj_train"],
            )
            config.experiment.config.logging.output_dir = str(
                output_dir_from_run_name(config)
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=tutorial/dist_agent_5lm_2obj_eval",
                    # We don't need to run the whole thing.
                    "experiment.config.n_eval_epochs=1",
                    "experiment.config.max_eval_steps=3",
                    "experiment.config.max_total_steps=3",
                ],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

    def test_omniglot_training_and_inference(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=["experiment=tutorial/omniglot_training"],
            )
            inference_output_dir = str(output_dir_from_run_name(config))
            config.experiment.config.logging.output_dir = inference_output_dir
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

            config = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=tutorial/omniglot_inference",
                    f"experiment.config.model_name_or_path={inference_output_dir}/pretrained/",
                    # We don't need to run the whole thing.
                    "experiment.config.n_eval_epochs=1",
                    "experiment.config.max_eval_steps=3",
                    "experiment.config.max_total_steps=3",
                ],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()

    def test_monty_meets_world_2dimage_inference(self):
        with hydra.initialize(version_base=None, config_path="../../conf"):
            config = hydra.compose(
                config_name="experiment",
                overrides=[
                    "experiment=tutorial/monty_meets_world_2dimage_inference",
                    # Non-interactive
                    "experiment.config.logging.wandb_handlers=[]",
                    "experiment.config.show_sensor_output=false",
                    # We don't need to run the whole thing.
                    "experiment.config.n_eval_epochs=1",
                    "experiment.config.max_eval_steps=3",
                    "experiment.config.max_total_steps=3",
                    "experiment.config.monty_config.monty_args.num_exploratory_steps=3",
                ],
            )
            experiment = hydra.utils.instantiate(config.experiment)
            with experiment:
                experiment.run()
