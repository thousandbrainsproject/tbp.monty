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
import torch

from tbp.monty.frameworks.environments.embodied_data import SaccadeOnImageDataLoader

from .monty_experiment import MontyExperiment

logger = logging.getLogger(__name__)


class MontyObjectRecognitionExperiment(MontyExperiment):
    """Experiment customized for object-pose recognition with a single object.

    Adds additional logging of the target object and pose for each episode and
    specific terminal states for object recognition. It also adds code for
    handling a matching and an exploration phase during each episode when training.

    Note that this experiment assumes a particular model configuration, in order
    for the show_observations method to work: a zoomed out "view_finder"
    rgba sensor and an up-close "patch" depth sensor
    """

    def run_episode(self):
        """Episode that checks the terminal states of an object recognition episode."""
        self.pre_episode()
        last_step = self.run_episode_steps()
        self.post_episode(last_step)

    def pre_episode(self):
        """Pre-episode hook.

        Pre episode where we pass the primary target object, as well as the mapping
        between semantic ID to labels, both for logging/evaluation purposes.
        """
        # TODO, eventually it would be better to pass
        # self.dataloader.semantic_id_to_label via an "Observation" object when this is
        # eventually implemented, such that we can ensure this information is never
        # inappropriately accessed and used
        if hasattr(self.dataloader, "semantic_id_to_label"):
            self.model.pre_episode(
                self.dataloader.primary_target, self.dataloader.semantic_id_to_label
            )
        else:
            self.model.pre_episode(self.dataloader.primary_target)
        self.dataloader.pre_episode()

        self.max_steps = self.max_train_steps
        if self.model.experiment_mode != "train":
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)

        if self.show_sensor_output:
            self.initialize_online_plotting()

    def run_episode_steps(self):
        """Runs one episode of the experiment.

        At each step, observations are collected from the dataloader and either passed
        to the model or sent directly to the motor system. We also check if a terminal
        condition was reached at each step and increment step counters.

        Returns:
            The number of total steps taken in the episode.
        """
        for loader_step, observation in enumerate(self.dataloader):
            if self.show_sensor_output:
                self.show_observations(observation, loader_step)

            if self.model.check_reached_max_matching_steps(self.max_steps):
                logger.info(
                    f"Terminated due to maximum matching steps : {self.max_steps}"
                )
                # Need to break here already, otherwise there are problems
                # when the object is recognized in the last step
                return loader_step

            if loader_step >= (self.max_total_steps):
                logger.info(f"Terminated due to maximum episode steps : {loader_step}")
                self.model.deal_with_time_out()
                return loader_step

            if self.model.is_motor_only_step:
                logger.debug(
                    "Performing a motor-only step, so passing info straight to motor"
                )
                # On these sensations, we just want to pass information to the motor
                # system, so bypass the main model step (i.e. updating of LMs)
                self.model.pass_features_directly_to_motor_system(observation)
            else:
                self.model.step(observation)

            if self.model.is_done:
                # Check this right after step to avoid setting time out
                # after object was already recognized.
                return loader_step
        # handle case where spiral policy calls StopIterator in motor policy
        self.model.set_is_done()
        return loader_step


class MontyGeneralizationExperiment(MontyObjectRecognitionExperiment):
    """Remove the tested object model from memory to see what is recognized instead."""

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        if "model.pt" not in self.model_path:
            model_path = os.path.join(self.model_path, "model.pt")
        state_dict = torch.load(model_path)
        print(f"loading models again from {model_path}")
        self.model.load_state_dict(state_dict)
        super().pre_episode()
        target_object = self.dataloader.primary_target["object"]
        print(f"removing {target_object}")
        for lm in self.model.learning_modules:
            lm.graph_memory.remove_graph_from_memory(target_object)
            print(f"graphs in memory: {lm.get_all_known_object_ids()}")
