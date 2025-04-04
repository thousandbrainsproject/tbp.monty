# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching import (
    EvidenceGraphLM,
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.states import State


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
        self.gsg.wait_growth_multiplier = 1

    def reset(self) -> None:
        super().reset()
        self.evidence = {}
        self.last_location = {}

    def _add_displacements(self, obs: List[State]) -> Tuple[List[State], bool]:
        """Add displacements to the current observation.

        For each input channel, this function computes the displacement vector by
        subtracting the current location from the last observed location. It then
        updates `self.last_location` for use in the next step. If any observation
        has a recorded previous location, we assume movement has occurred.

        In this unsupervised inference setting, the displacement is set to zero
        at the beginning of the first episode when the last location is not set.

        Args:
            obs (List[State]): A list of observations to which displacements will be
                added.

        Returns:
            - obs (List[State]): The list of observations, each updated with a
                displacement vector.
            - not_moved (bool): True if none of the observations had a prior location
              (i.e., no movement detected), False otherwise.
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

    def matching_step(self, observations: List[State]) -> None:
        """Perform the matching step given a list of observations.

        This updates the displacement buffer, computes possible matches,
        steps the goal state generator. Additionally, the stats for the current
        step are saved in the buffer.

        Args:
            observations (List[State]): The current step's observations.
        """
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

    def _add_detailed_stats(
        self, stats: Dict[str, Any], get_rotations: bool
    ) -> Dict[str, Any]:
        """Add detailed statistics to the logging dictionary.

        This includes metrics like the max evidence score per object, the theoretical
        limit of Monty i.e., pose error of Monty's best hypothesis on the target object)
        , and the pose error of the MLH hypothesis on the target object.

        Args:
            stats (Dict[str, Any]): The existing statistics dictionary to augment.
            get_rotations (bool): Flag indicating if rotation stats are needed
                                  (currently unused in implementation).

        Returns:
            Dict[str, Any]: Updated statistics dictionary.
        """
        stats["max_evidence"] = {k: max(v) for k, v in self.evidence.items()}
        stats["target_object_theoretical_limit"] = (
            self._target_object_theoretical_limit()
        )
        stats["target_object_pose_error"] = self._target_object_pose_error()
        return stats

    def _target_object_theoretical_limit(self) -> float:
        """Compute the theoretical minimum rotation error on the target object.

        This considers all possible hypotheses rotations on the target object
        and compares them to the target's rotation.

        Returns:
            float: The minimum achievable rotation error (in radians).
        """
        poses = self.possible_poses[self.primary_target].copy()
        hyp_rotations = Rotation.from_matrix(poses).inv().as_quat().tolist()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)

        min_error = np.pi
        for rot in hyp_rotations:
            rot = Rotation.from_quat(rot)
            error = (rot * target_rotation.inv()).magnitude()
            min_error = min(min_error, error)

        return min_error

    def _target_object_pose_error(self) -> float:
        """Compute the actual rotation error between predicted and target pose.

        This compares the most likely hypothesis pose (based on evidence) on the target
        object with the ground truth rotation of the target object.

        Returns:
            float: The rotation error (in radians).
        """
        target_rot = Rotation.from_quat(self.primary_target_rotation_quat)
        obj_mlh_id = np.argmax(self.evidence[self.primary_target])
        obj_rotation = Rotation.from_matrix(
            self.possible_poses[self.primary_target][obj_mlh_id]
        )

        error = (obj_rotation * target_rot.inv()).magnitude()
        return error
