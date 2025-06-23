# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging

import numpy as np

from tbp.monty.frameworks.models.graph_matching import (
    MontyForGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)


class MontyForEvidenceGraphMatching(MontyForGraphMatching):
    """Monty model for evidence based graphs.

    Customize voting and union of possible matches.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

    def _pass_infos_to_motor_system(self):
        """Pass processed observations and goal-states to the motor system.

        Currently there is no complex connectivity or hierarchy, and all generated
        goal-states are considered bound for the motor-system. TODO M change this.
        """
        super()._pass_infos_to_motor_system()

        # Check the motor-system can receive goal-states
        if self.motor_system._policy.use_goal_state_driven_actions:
            best_goal_state = None
            best_goal_confidence = -np.inf
            for current_goal_state in self.gsg_outputs:
                if (
                    current_goal_state is not None
                    and current_goal_state.confidence > best_goal_confidence
                ):
                    best_goal_state = current_goal_state
                    best_goal_confidence = current_goal_state.confidence

            self.motor_system._policy.set_driving_goal_state(best_goal_state)

    def _combine_votes(self, votes_per_lm):
        """Combine evidence from different lms.

        Returns:
            The combined votes.
        """
        combined_votes = []
        for i in range(len(self.learning_modules)):
            lm_state_votes = {}
            if votes_per_lm[i] is not None:
                receiving_lm_pose = votes_per_lm[i]["sensed_pose_rel_body"]
                for j in self.lm_to_lm_vote_matrix[i]:
                    if votes_per_lm[j] is not None:
                        sending_lm_pose = votes_per_lm[j]["sensed_pose_rel_body"]
                        sensor_disp = np.array(receiving_lm_pose[0]) - np.array(
                            sending_lm_pose[0]
                        )
                        sensor_rotation_disp, _ = align_orthonormal_vectors(
                            sending_lm_pose[1:],
                            receiving_lm_pose[1:],
                            as_scipy=False,
                        )
                        logging.debug(
                            f"LM {j} to {i} - displacement: {sensor_disp}, "
                            f"rotation: "
                            f"{sensor_rotation_disp}"
                        )
                        for obj in votes_per_lm[j]["possible_states"].keys():
                            # Get the displacement between the sending and receiving
                            # sensor and take this into account when transmitting
                            # possible locations on the object.
                            # "If I am here, you should be there."
                            lm_states_for_object = votes_per_lm[j]["possible_states"][
                                obj
                            ]
                            # Take the location votes and transform them so they would
                            # apply to the receiving LMs sensor. Basically saying, if my
                            # sensor is here and in this pose then your sensor should be
                            # there in that pose.
                            # NOTE: rotation votes are not being used right now.
                            transformed_lm_states_for_object = []
                            for s in lm_states_for_object:
                                # need to make a copy because the same vote state may be
                                # transformed in different ways depending on the
                                # receiving LMs' pose
                                new_s = copy.deepcopy(s)
                                rotated_displacement = new_s.get_pose_vectors().dot(
                                    sensor_disp
                                )
                                new_s.transform_morphological_features(
                                    translation=rotated_displacement,
                                    rotation=sensor_rotation_disp,
                                )
                                transformed_lm_states_for_object.append(new_s)
                            if obj in lm_state_votes.keys():
                                lm_state_votes[obj].extend(
                                    transformed_lm_states_for_object
                                )
                            else:
                                lm_state_votes[obj] = transformed_lm_states_for_object
            logging.debug(f"VOTE from LMs {self.lm_to_lm_vote_matrix[i]} to LM {i}")
            vote = lm_state_votes
            combined_votes.append(vote)
        return combined_votes

    def switch_to_exploratory_step(self):
        """Switch to exploratory step.

        Also, set mlh evidence high enough to generate output during exploration.
        """
        super().switch_to_exploratory_step()
        # Make sure new object ID gets communicated to higher level LMs during
        # exploration.
        for lm in self.learning_modules:
            lm.current_mlh["evidence"] = lm.object_evidence_threshold + 1
