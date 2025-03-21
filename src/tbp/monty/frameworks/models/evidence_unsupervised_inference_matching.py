# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.models.evidence_matching import MontyForEvidenceGraphMatching


class MontyForUnsupervisedEvidenceGraphMatching(MontyForEvidenceGraphMatching):
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

            # These are left here as a sanity check.
            # When the buffer is reset, displacements become None (len(buffer) <=1),
            # which causes hypothesis space to be reinitialized and resample informed
            # hypotheses.

            # lm.buffer.reset()
            # lm.gsg.reset()
