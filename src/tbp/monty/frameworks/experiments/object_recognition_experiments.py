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

import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.environments.embodied_data import (
    SaccadeOnImageEnvironmentInterface,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.monty_experiment import (
    MontyExperiment,
)

__all__ = ["MontyGeneralizationExperiment", "MontyObjectRecognitionExperiment"]

logger = logging.getLogger(__name__)


class MontyObjectRecognitionExperiment(MontyExperiment):
    """Experiment customized for object-pose recognition with a single object.

    Adds additional logging of the target object and pose for each episode and
    specific terminal states for object recognition. It also adds code for
    handling a matching and an exploration phase during each episode when training.

    Note that this experiment assumes a particular model configuration in order
    for the show_observations method to work: a zoomed-out "view_finder"
    RGBA sensor and an up-close "patch" depth sensor.
    """

    def run_episode(self):
        """Episode that checks the terminal states of an object recognition episode."""
        self.pre_episode()
        last_step = self.run_episode_steps()
        self.post_episode(last_step)

    def pre_episode(self):
        """Pre-episode hook.

        Passes the primary target object and the mapping from semantic IDs to labels
        to the Monty model for logging and reporting evaluation results.
        """
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

        # TODO, eventually it would be better to pass
        # self.env_interface.semantic_id_to_label via an "Observation" object when this
        # is eventually implemented, such that we can ensure this information is never
        # inappropriately accessed and used
        if hasattr(self.env_interface, "semantic_id_to_label"):
            # TODO: Fix invalid pre_episode signature call
            self.model.pre_episode(
                self.rng,
                self.env_interface.primary_target,
                self.env_interface.semantic_id_to_label,
            )
        else:
            # TODO: Fix invalid pre_episode signature call
            self.model.pre_episode(self.rng, self.env_interface.primary_target)
        self.env_interface.pre_episode(self.rng)

        self.max_steps = self.max_train_steps
        if self.experiment_mode is not ExperimentMode.TRAIN:
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)

        if self.show_sensor_output:
            self.live_plotter.initialize_online_plotting()

    def run_episode_steps(self):
        """Runs one episode of the experiment.

        At each step, observations are collected from the env_interface and either
        passed to the model or sent directly to the motor system. We also check if a
        terminal condition was reached at each step and increment step counters.

        Returns:
            The number of total steps taken in the episode.
        """
        step = 0
        ctx = RuntimeContext(rng=self.rng)
        while True:
            try:
                observations = self.env_interface.step(ctx, first=(step == 0))
            except StopIteration:
                # TODO: StopIteration is being thrown by NaiveScanPolicy to signal
                #       episode termination. This is a holdover from when we used
                #       iterators. However, this also abdicates control of the
                #       experiment to the policy. We should find a better way to handle
                #       this, so that the experiment can control the episode termination
                #       fully. For example, we know how many steps the policy will take,
                #       so the experiment can set max steps based on that knowledge
                #       alone.
                self.model.set_is_done()
                return step

            if self.show_sensor_output:
                is_saccade_on_image_data_loader = isinstance(
                    self.env_interface, SaccadeOnImageEnvironmentInterface
                )
                self.live_plotter.show_observations(
                    *self.live_plotter.hardcoded_assumptions(observations, self.model),
                    step,
                    is_saccade_on_image_data_loader,
                )

            if self.model.check_reached_max_matching_steps(self.max_steps):
                logger.info(
                    f"Terminated due to maximum matching steps : {self.max_steps}"
                )
                # Need to break here already, otherwise there are problems
                # when the object is recognized in the last step
                return step

            if step >= (self.max_total_steps):
                logger.info(f"Terminated due to maximum episode steps : {step}")
                self.model.deal_with_time_out()
                return step

            if self.model.is_motor_only_step:
                logger.debug(
                    "Performing a motor-only step, so passing info straight to motor"
                )
                # On these sensations, we just want to pass information to the motor
                # system, so bypass the main model step (i.e. updating of LMs)
                self.model.pass_features_directly_to_motor_system(observations)
            else:
                self.model.step(observations)

            if self.model.is_done:
                # Check this right after step to avoid setting time out
                # after object was already recognized.
                return step

            step += 1


class MontyGeneralizationExperiment(MontyObjectRecognitionExperiment):
    """Remove the tested object model from memory to see what is recognized instead."""

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        if "model.pt" not in self.model_path.parts:
            model_path = self.model_path / "model.pt"
        state_dict = torch.load(model_path)
        print(f"loading models again from {model_path}")
        self.model.load_state_dict(state_dict)
        super().pre_episode()
        target_object = self.env_interface.primary_target["object"]
        print(f"removing {target_object}")
        for lm in self.model.learning_modules:
            lm.graph_memory.remove_graph_from_memory(target_object)
            print(f"graphs in memory: {lm.get_all_known_object_ids()}")
