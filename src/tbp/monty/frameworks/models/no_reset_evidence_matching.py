# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import numpy as np

from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)


class MontyForNoResetEvidenceGraphMatching(MontyForEvidenceGraphMatching):
    """Monty class for unsupervised inference without explicit episode resets.

    This variant of `MontyForEvidenceGraphMatching` is designed for unsupervised
    inference experiments where objects may change dynamically without any reset
    signal. Unlike standard experiments, this class avoids resetting Monty's
    internal state (e.g., hypothesis space, evidence scores) between episodes.

    This setup better reflects real-world conditions, where object boundaries
    are ambiguous and no supervisory signal is available to indicate when a new
    object appears. Only minimal state — such as step counters and termination
    flags — is reset to prevent buffers from accumulating across objects. Additionally,
    Monty is currently forced to switch to Matching state. Evaluation of unsupervised
    inference is performed over a fixed number of matching steps per object.

    *Intended for evaluation-only runs using pre-trained models, with Monty
    remaining in the matching phase throughout.*
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track whether `pre_episode` has been called at least once.
        # There are two separate issues this helps avoid:
        #
        # 1. Some internal variables in SMs and LMs (e.g., `stepwise_targets_list`,
        #    `terminal_state`, `is_exploring`, `visited_locs`) are not initialized
        #    in `__init__`, but only inside `pre_episode`. Ideally, these should be
        #    initialized once in `__init__` and reset in `pre_episode`, but fixing
        #    this would require changes across multiple classes.
        #
        # 2. The order of operations: Graphs are loaded into LMs *after* the Monty
        #    object is constructed but *before* `pre_episode` is called. Some
        #    functions (e.g., in `EvidenceGraphLM`) depend on the graph being loaded to
        #    compute initial possible matches inside `pre_episode`, and this cannot
        #    be safely moved into `__init__`.
        #
        # As a workaround, we allow `pre_episode` to run normally once (to complete
        # required initialization), and skip full resets on subsequent calls.
        # TODO: Remove initialization logic from `pre_episode`
        self.init_pre_episode = False

    def pre_episode(self, primary_target, semantic_id_to_label=None):
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
        """Resets buffers for LMs and SMs."""
        for lm in self.learning_modules:
            lm.buffer.reset()
        for sm in self.sensor_modules:
            sm.raw_observations = []
            sm.sm_properties = []
            sm.processed_obs = []


class NoResetEvidenceGraphLM(EvidenceGraphLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_location = {}

        # it does not make sense for the wait factor to exponentially
        # grow when objects are swapped without any supervisory signal.
        self.gsg.wait_growth_multiplier = 1

    def reset(self):
        super().reset()
        self.evidence = {}
        self.last_location = {}

    def _add_displacements(self, obs):
        """Add displacements to the current observation.

        For each input channel, this function computes the displacement vector by
        subtracting the current location from the last observed location. It then
        updates `self.last_location` for use in the next step. If any observation
        has a recorded previous location, we assume movement has occurred.

        In this unsupervised inference setting, the displacement is set to zero
        at the beginning of the first episode when the last location is not set.

        Args:
            obs: A list of observations to which displacements will be added.

        Returns:
            - obs: The list of observations, each updated with a displacement vector.
        """
        for o in obs:
            if o.sender_id in self.last_location.keys():
                displacement = o.location - self.last_location[o.sender_id]
            else:
                displacement = np.zeros(3)
            o.set_displacement(displacement)
            self.last_location[o.sender_id] = o.location
        return obs

    def _agent_moved_since_reset(self):
        """Overwrites the logic of whether the agent has moved since the last reset.

        In unsupervised inference, the first movement is detected on the first
        episode only. If a `last_location` exists, then first movement has occurred.

        Returns:
            - Whether the agent has moved since the last reset.
        """
        return len(self.last_location) > 0
