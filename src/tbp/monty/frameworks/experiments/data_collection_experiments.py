# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .object_recognition_experiments import MontyObjectRecognitionExperiment

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()


class DataCollectionExperiment(MontyObjectRecognitionExperiment):
    """Collect data in environment without performing inference.

    Stripped down experiment, to explore points on the object and save JUST the
    resulting observations as a .pt file. This was used to collect data that can then
    be used offline to quickly test other, non-Monty methods (like ICP). Mostly useful
    for methods that require batches of observations and do not work with inference
    through movement over the object. Otherwise would recommend to implement approaches
    directly in the Monty framework instead of using offline data.
    """

    def run_episode(self):
        """Episode that checks the terminal states of an object recognition episode."""
        self.pre_episode()
        for step, observation in tqdm(enumerate(self.dataloader)):
            if step > self.max_steps:
                break
            if self.show_sensor_output:
                self.show_observations(observation, step)
            self.pass_features_to_motor_system(observation, step)
        self.post_episode()

    def pass_features_to_motor_system(self, observation, step):
        self.model.aggregate_sensory_inputs(observation)
        self.model.motor_system.processed_observations = (
            self.model.sensor_module_outputs[0]
        )
        # Add the object and action to the observation dict
        self.model.sensor_modules[0].processed_obs[-1]["object"] = (
            self.dataloader.primary_target["object"]
        )
        self.model.sensor_modules[0].processed_obs[-1]["action"] = (
            None
            if self.model.motor_system.action is None
            else (
                f"{self.model.motor_system.action.agent_id}."
                f"{self.model.motor_system.action.name}"
            )
        )
        # Only include observations coming right before a move_tangentially action
        if step > 0 and (
            self.model.motor_system.action is None
            or self.model.motor_system.action.name != "move_tangentially"
        ):
            del self.model.sensor_modules[0].processed_obs[-2]

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        self.model.pre_episode()
        self.dataloader.pre_episode()
        self.max_steps = self.max_train_steps
        self.logger_handler.pre_episode(self.logger_args)
        if self.show_sensor_output:
            self.initialize_online_plotting()

    def post_episode(self):
        torch.save(
            self.model.sensor_modules[0].processed_obs[:-1],
            os.path.join(self.output_dir, f"observations{self.train_episodes}.pt"),
        )
        self.dataloader.post_episode()
        self.train_episodes += 1

    def post_epoch(self):
        # This stripped down expt only allows for one pass
        pass
