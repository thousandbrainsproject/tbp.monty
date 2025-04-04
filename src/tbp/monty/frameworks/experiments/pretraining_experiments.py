# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict

from .monty_experiment import MontyExperiment


class MontySupervisedObjectPretrainingExperiment(MontyExperiment):
    """Just run exploratory steps and tell the model the object and pose.

    This is used to pretrain models of objects such that we don't always have to start
    learning from scratch. Since we provide the model with the object label and pose we
    do not have to perform matching steps. We just run exploratory steps and use the
    collected observations to update the models in memory.

    NOTE: This is not really an experiment, it is more a pretraining step to generate
    models that can then be loaded at the beginning of an experiment.
    """
    def __init__(self, config):
        # If we just add "pretrained" to dir at save time, then logs are stored in one
        # place and models in another. Changing the config ensures every reference to
        # output_dir has "pretrained" added to it
        config = config_to_dict(config)
        output_dir = config["logging_config"]["output_dir"]
        config["logging_config"]["output_dir"] = os.path.join(output_dir, "pretrained")
        super().__init__(config)

    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.sensor_pos = np.array(
            config["dataset_args"]["env_init_args"]["agents"][0]["agent_args"][
                "positions"
            ]
        )

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
        # set is_seeking_match False and goes in exploratory mode
        self.model.step_type = "exploratory_step"
        # Pass target info to model
        target = self.dataloader.primary_target
        self.model.detected_object = self.model.primary_target["object"]
        for lm in self.model.learning_modules:
            lm.detected_object = target["object"]
            lm.buffer.stats["possible_matches"] = [target["object"]]
            lm.buffer.stats["detected_location_on_model"] = np.array(target["position"])
            lm.buffer.stats["detected_location_rel_body"] = np.array(target["position"])
            lm.buffer.stats["detected_rotation"] = target["euler_rotation"]
            lm.detected_rotation_r = Rotation.from_quat(target["quat_rotation"]).inv()
            lm.buffer.stats["detected_scale"] = target["scale"]
        # Collect data about the object (exploratory steps)
        num_steps = 0
        for observation in self.dataloader:
            num_steps += 1
            self.model.step(observation)
            if self.model.is_done:
                break

            # Even if many exploratory steps have not sent information to learning
            # modules (so is_done remains False), eventually terminate exploration
            # TODO: should we use model.total_steps here?
            if self.model.episode_steps >= self.max_total_steps:
                break
        if len(self.model.learning_modules) > 1:
            for i, lm in enumerate(self.model.learning_modules):
                if i == 0:
                    first_pos = self.sensor_pos[0]
                else:
                    lm_offset = self.sensor_pos[i] - first_pos

                    lm.buffer.stats["detected_location_rel_body"] += lm_offset
                    # Rotate offset into model RF
                    lm_offset_model_rf = lm.detected_rotation_r.apply(lm_offset)
                    lm.buffer.stats["detected_location_on_model"] += lm_offset_model_rf

        # Update the model in memory
        self.post_episode(num_steps)

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        self.model.pre_episode(self.dataloader.primary_target)
        self.dataloader.pre_episode()

        self.max_steps = self.max_train_steps  # no eval mode here

        self.logger_handler.pre_episode(self.logger_args)

    def post_epoch(self):
        """Post epoch without saving state_dict."""
        self.logger_handler.post_epoch(self.logger_args)

        self.train_epochs += 1
        self.train_dataloader.post_epoch()

    def train(self):
        """Save state_dict at the end of training."""
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
        logging.warning("No evalualtion mode for supervised experiment.")
        pass
