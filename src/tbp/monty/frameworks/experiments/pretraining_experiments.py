# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environments.embodied_data import (
    SaccadeOnImageEnvironmentInterface,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import (
    MontyExperiment,
)

__all__ = ["MontySupervisedObjectPretrainingExperiment"]

logger = logging.getLogger(__name__)


class MontySupervisedObjectPretrainingExperiment(MontyExperiment):
    """Just run exploratory steps and tell the model the object and pose.

    This is used to pretrain models of objects such that we don't always have to start
    learning from scratch. Since we provide the model with the object label and pose we
    do not have to perform matching steps. We just run exploratory steps and use the
    collected observations to update the models in memory.

    NOTE: This is not really an experiment, it is more a pretraining step to generate
    models that can then be loaded at the beginning of an experiment.
    """

    def __init__(self, config: DictConfig):
        # If we just add "pretrained" to dir at save time, then logs are stored in one
        # place and models in another. Changing the config ensures every reference to
        # output_dir has "pretrained" added to it
        config = OmegaConf.to_object(config)
        output_dir = Path(config["logging"]["output_dir"])
        config["logging"]["output_dir"] = output_dir / "pretrained"
        self.first_epoch_object_location = {}
        super().__init__(config)

    def setup_experiment(self, config):
        super().setup_experiment(config)
        if "agents" in config["env_interface_config"]["env_init_args"].keys():
            self.sensor_pos = np.array(
                config["env_interface_config"]["env_init_args"]["agents"]["agent_args"][
                    "positions"
                ]
            )
        else:
            self.sensor_pos = np.array([0, 0, 0])

    def run_episode(self):
        """Run a supervised episode on one object in one pose.

        In a supervised episode we only make exploratory steps (no object recognition
        is attempted) since the target label is provided. The target label and pose
        is then used to update the object model in memory.
        This can for instance be used to warm-up the training by starting with some
        models in memory instead of completely from scatch. It also makes testing
        easier as long as we don't have a good solution to dealing with incomplete
        objects.
        """
        self.pre_episode()
        # Save compute if we are providing labels to all models, so don't need to
        # perform matching parts of LM updates (default is matching_step)
        all_lm_ids = [lm.learning_module_id for lm in self.model.learning_modules]
        if set(self.supervised_lm_ids) == set(all_lm_ids):
            self.model.switch_to_exploratory_step()

        # Collect data about the object (exploratory steps)
        num_steps = 0
        for observation in self.env_interface:
            num_steps += 1
            if self.show_sensor_output:
                is_saccade_on_image_env_interface = isinstance(
                    self.env_interface, SaccadeOnImageEnvironmentInterface
                )
                self.live_plotter.show_observations(
                    *self.live_plotter.hardcoded_assumptions(observation, self.model),
                    num_steps,
                    is_saccade_on_image_env_interface,
                )
            self.model.step(observation)
            if self.model.is_done:
                break

            # Even if many exploratory steps have not sent information to learning
            # modules (so is_done remains False), eventually terminate exploration
            # TODO: should we use model.total_steps here?
            if self.model.episode_steps >= self.max_total_steps:
                break

        # Pass target info to model --> will overwrite (where specified)
        # with the ground truth labels just before models are updated in memory.
        target = self.env_interface.primary_target
        self.model.detected_object = self.model.primary_target["object"]
        for lm in self.model.learning_modules:
            if lm.learning_module_id in self.supervised_lm_ids:
                lm.detected_object = target["object"]
                lm.buffer.stats["possible_matches"] = [target["object"]]
                lm.buffer.stats["detected_location_on_model"] = (
                    self.first_epoch_object_location[target["object"]]
                )
                lm.buffer.stats["detected_location_rel_body"] = np.array(
                    target["position"]
                )
                lm.buffer.stats["detected_rotation"] = target["euler_rotation"]
                lm.detected_rotation_r = Rotation.from_quat(
                    target["quat_rotation"]
                ).inv()
                lm.buffer.stats["detected_scale"] = target["scale"]
            else:
                # wipe LMs hypotheses so we don't update those models
                lm.possible_matches = []
                lm.detected_object = None
                lm.detected_pose = None
                lm.detected_rotation_r = None
                lm.buffer.stats["detected_location_on_model"] = None
                lm.buffer.stats["detected_location_rel_body"] = None
                lm.buffer.stats["detected_rotation"] = None
                lm.buffer.stats["detected_scale"] = None

        if len(self.model.learning_modules) > 1:
            for i, lm in enumerate(self.model.learning_modules):
                if i == 0:
                    first_pos = self.sensor_pos[0]
                else:
                    lm_offset = self.sensor_pos[i] - first_pos
                    if lm.detected_object:
                        lm.buffer.stats["detected_location_rel_body"] += lm_offset
                        # Rotate offset into model RF
                        lm_offset_model_rf = lm.detected_rotation_r.apply(lm_offset)
                        lm.buffer.stats["detected_location_on_model"] += (
                            lm_offset_model_rf
                        )

        # Update the model in memory
        self.post_episode(num_steps)

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        if self.experiment_mode is ExperimentMode.TRAIN:
            logger.info(
                f"running train epoch {self.train_epochs} "
                f"train episode {self.train_episodes}"
            )
        else:
            logger.info(
                f"running eval epoch {self.eval_epochs} "
                f"eval episode {self.eval_episodes}"
            )

        self.reset_episode_rng()

        # TODO: Fix invalid pre_episode signature call
        self.model.pre_episode(self.rng, self.env_interface.primary_target)
        self.env_interface.pre_episode(self.rng)

        self.max_steps = self.max_train_steps  # no eval mode here

        self.logger_handler.pre_episode(self.logger_args)

        # if it's the first time this object is shown, save it's location. This is
        # needed to provide the correct offset from the learned model when supervising.
        current_object = self.env_interface.primary_target["object"]
        if current_object not in self.first_epoch_object_location:
            self.first_epoch_object_location[current_object] = (
                self.env_interface.primary_target["position"]
            )

        if self.show_sensor_output:
            self.live_plotter.initialize_online_plotting()

    def post_epoch(self):
        """Post epoch without saving state_dict."""
        self.logger_handler.post_epoch(self.logger_args)

        self.train_epochs += 1
        self.train_env_interface.post_epoch()

    def train(self):
        """Save state_dict at the end of training."""
        self.experiment_mode = ExperimentMode.TRAIN
        self.logger_handler.pre_train(self.logger_args)
        self.model.set_experiment_mode("train")
        for sm in self.model.sensor_modules:
            sm.save_raw_obs = False
        for _ in range(self.n_train_epochs):
            self.run_epoch()
        self.logger_handler.post_train(self.logger_args)
        # Save only at the end of pretraining
        self.save_state_dict(output_dir=self.output_dir)

    def evaluate(self):
        """Use experiment just for supervised pretraining -> no eval."""
        logger.warning("No evalualtion mode for supervised experiment.")
        pass
