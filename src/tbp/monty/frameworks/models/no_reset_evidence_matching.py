# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np

from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)


class MontyForNoResetEvidenceGraphMatching(MontyForEvidenceGraphMatching):
    """This class removes explicit reset logic in unsupervised inference experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This keeps track of whether the first `pre_episode` is called.
        # The experiment will not run unless the first `pre_episode` function
        # is called, which suggests that some class initialization code is
        # inside `pre_episode`. Initialization and resetting code should be separate.
        # TODO: Remove initialization logic from `pre_episode`
        self.init_pre_episode = False

    def pre_episode(self, primary_target, semantic_id_to_label=None):
        # call the parent `pre_episode` at the beginning of the experiment
        if not self.init_pre_episode:
            self.init_pre_episode = True
            return super().pre_episode(primary_target, semantic_id_to_label)

        # reset terminal state
        self._is_done = False
        self.reset_episode_steps()
        self.switch_to_matching_step()

        # keep target up-to-date for logging
        self.primary_target = primary_target
        self.semantic_id_to_label = semantic_id_to_label
        for lm in self.learning_modules:
            lm.primary_target = primary_target["object"]
            lm.primary_target_rotation_quat = primary_target["quat_rotation"]

        # reset LMs and SMs buffers to save memory
        self._reset_modules_buffers()

    def _reset_modules_buffers(self):
        for lm in self.learning_modules:
            lm.buffer.reset()
            lm.gsg.wait_factor = 1
        for sm in self.sensor_modules:
            sm.raw_observations = []
            sm.sm_properties = []
            sm.processed_obs = []


class NoResetEvidenceGraphLM(EvidenceGraphLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_location = {}

    def reset(self):
        super().reset()
        self.evidence = {}
        self.last_location = {}

    def _add_displacements(self, obs):
        """Add displacements to the current observation.

        The observation consists of features at a location. To get the displacement we
        have to look at the previous observation stored in the buffer.

        Args:
            obs: Observations to add displacements to.

        Returns:
            Observations with displacements.
        """
        not_moved = True
        for o in obs:
            if o.sender_id in self.last_location.keys():
                displacement = o.location - self.last_location[o.sender_id]
                not_moved = False
            else:
                displacement = np.zeros(3)
            o.set_displacement(displacement)
            self.last_location[o.sender_id] = o.location
        return obs, not_moved

    def matching_step(self, observations):
        """Update the possible matches given an observation."""
        buffer_data, not_moved = self._add_displacements(observations)
        self.buffer.append(buffer_data)
        self.buffer.append_input_states(observations)

        if not_moved:
            logging.debug("we have not moved yet.")
        else:
            logging.debug("performing matching step.")

        self._compute_possible_matches(observations, not_moved=not_moved)

        if len(self.get_possible_matches()) == 0:
            self.set_individual_ts(terminal_state="no_match")

        self.gsg.step_gsg(observations)

        stats = self.collect_stats_to_save()
        self.buffer.update_stats(stats, append=self.has_detailed_logger)
