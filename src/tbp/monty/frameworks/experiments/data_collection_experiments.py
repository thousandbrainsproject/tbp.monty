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

import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

logger = logging.getLogger(__name__)


class DataCollectionExperiment(MontyObjectRecognitionExperiment):
    """Collect data in environment without performing inference.

    Stripped-down experiment to explore points on the object and save the resulting
    observations as a .pt file. This can be used to collect data that can then be used
    offline to quickly test other, non-Monty methods (like ICP). It is mostly useful for
    methods that require batches of observations and do not work with inference through
    movement over the object. Otherwise, we recommend implementing approaches directly
    in the Monty framework rather than using offline data.
    """

    def run_episode(self):
        self.pre_episode()
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
                break

            if step > self.max_steps:
                break
            if self.show_sensor_output:
                self.live_plotter.show_observations(
                    *self.live_plotter.hardcoded_assumptions(observations, self.model),
                    step,
                )
            self.pass_features_to_motor_system(ctx, observations, step)
            step += 1

        self.post_episode()

    def pass_features_to_motor_system(self, ctx: RuntimeContext, observation, step):
        self.model.aggregate_sensory_inputs(ctx, observation)
        self.model.motor_system._policy.processed_observations = (
            self.model.sensor_module_outputs[0]
        )
        # Add the object and action to the observation dict
        self.model.sensor_modules[0].processed_obs[-1]["object"] = (
            self.env_interface.primary_target["object"]
        )
        action_strings = [
            f"{action.agent_id}.{action.name}"
            for action in self.model.motor_system._policy.actions
        ]
        self.model.sensor_modules[0].processed_obs[-1]["actions"] = action_strings
        # Only include observations coming right before a move_tangentially action
        if step > 0 and (
            not self.model.motor_system._policy.actions
            or self.model.motor_system._policy.actions[0].name != "move_tangentially"
        ):
            del self.model.sensor_modules[0].processed_obs[-2]

    def pre_episode(self):
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

        self.model.pre_episode()
        self.env_interface.pre_episode(self.rng)
        self.max_steps = self.max_train_steps
        self.logger_handler.pre_episode(self.logger_args)
        if self.show_sensor_output:
            self.live_plotter.initialize_online_plotting()

    def post_episode(
        self,
        steps,  # noqa: ARG002
    ):
        torch.save(
            self.model.sensor_modules[0].processed_obs[:-1],
            self.output_dir / f"observations{self.train_episodes}.pt",
        )
        self.env_interface.post_episode()
        self.train_episodes += 1

    def post_epoch(self):
        # This stripped-down experiment only allows for one epoch.
        pass
