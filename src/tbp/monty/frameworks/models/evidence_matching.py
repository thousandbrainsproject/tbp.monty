# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging
import threading
import time

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.graph_matching import (
    GraphLM,
    GraphMemory,
    MontyForGraphMatching,
)
from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    GridObjectModel,
    GridTooSmallError,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.graph_matching_utils import (
    add_pose_features_to_tolerances,
    get_custom_distances,
    get_initial_possible_poses,
    get_relevant_curvature,
    get_scaled_evidences,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
    align_orthonormal_vectors,
    get_angles_for_all_hypotheses,
    get_more_directions_in_plane,
    rotate_pose_dependent_features,
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
        if self.motor_system.use_goal_state_driven_actions:
            best_goal_state = None
            best_goal_confidence = -np.inf
            for current_goal_state in self.gsg_outputs:
                if (
                    current_goal_state is not None
                    and current_goal_state.confidence > best_goal_confidence
                ):
                    best_goal_state = current_goal_state
                    best_goal_confidence = current_goal_state.confidence

            self.motor_system.set_driving_goal_state(best_goal_state)

    def _combine_votes(self, votes_per_lm):
        """Combine evidence from different lms.

        Returns:
            The combined votes.
        """
        combined_votes = []
        for i in range(len(self.learning_modules)):
            lm_state_votes = dict()
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


class EvidenceGraphLM(GraphLM):
    """Learning module that accumulates evidence for objects and poses.

    Matching Attributes:
        max_match_distance: Maximum distance of a tested and stored location to
            be matched.
        tolerances: How much can each observed feature deviate from the stored
            features to still be considered a match.
        feature_weights: How much should each feature be weighted when calculating
            the evidence update for hypotheses. Weights are stored in a dictionary with
            keys corresponding to features (same as keys in tolerances)
        feature_evidence_increment: Feature evidence (between 0 and 1) is multiplied
            by this value before being added to the overall evidence of a hypothesis.
            This factor is only multiplied with the feature evidence (not the pose
            evidence as opposed to the present_weight).
        max_nneighbors: Maximum number of nearest neighbors to consider in the
            radius of a hypothesis for calculating the evidence.
        initial_possible_poses: initial possible poses that should be tested for.
            In ["uniform", "informed", list]. default = "informed".
        evidence_update_threshold: How to decide which hypotheses should be updated.
            When this parameter is either '[int]%' or 'x_percent_threshold', then
            this parameter is applied to the evidence for the Most Likely Hypothesis
            (MLH) to determine a minimum evidence threshold in order for other
            hypotheses to be updated. Any hypotheses falling below the resulting
            evidence threshold do not get updated. The other options set a fixed
            threshold that does not take MLH evidence into account. In [int, float,
            '[int]%', 'mean', 'median', 'all', 'x_percent_threshold'].
        vote_evidence_threshold: Only send votes that have a scaled evidence above
            this threshold. Vote evidences are in the range of [-1, 1] so the threshold
            should not be outside this range.
        past_weight: How much should the evidence accumulated so far be weighted
            when combined with the evidence from the most recent observation.
        present_weight: How much should the current evidence be weighted when added
            to the previous evidence. If past_weight and present_weight add up to 1,
            the evidence is bounded and can't grow infinitely. NOTE: right now this
            doesn't give as good performance as with unbounded evidence since we don't
            keep a full history of what we saw. With a more efficient policy and better
            parameters that may be possible to use though and could help when moving
            from one object to another and to generally make setting thresholds etc.
            more intuitive.
        vote_weight: Vote evidence (between -1 and 1) in multiplied by this  value
            when being added to the overall evidence of a hypothesis. If past and
            current_weight add up to 1, it is use as weight in np.average to keep
            the evidence in a fixed range.

    Terminal Condition Attributes:
        object_evidence_threshold: Minimum required evidence for an object to be
            recognized. We additionally check that the evidence for this object is
            significantly higher than for all other objects.
        x_percent_threshold: Used in two places:
            1) All objects whose highest evidence is greater than the most likely
                objects evidence - x_percent of the most like objects evidence are
                considered possible matches. That means to only have one possible match,
                no other object can have more evidence than the candidate match's
                evidence - x percent of it.
            2) Within one object, possible poses are considered possible if their
                evidence is larger than the most likely pose of this object - x percent
                of this poses evidence.
            # TODO: should we use a separate theshold for within and between objects?
            If this value is larger, the model is usually more robust to noise and
            reaches a better performance but also requires a lot more steps to reach a
            terminal condition, especially if there are many similar object in the data
            set.
        path_similarity_threshold: How similar do paths have to be to be
            considered the same in the terminal condition check.
        pose_similarity_threshold: difference between two poses to be considered
            unique when checking for the terminal condition (in radians).
        required_symmetry_evidence: number of steps with unchanged possible poses
            to classify an object as symetric and go into terminal condition.

    Model Attributes:
        graph_delta_thresholds: Thresholds used to compare nodes in the graphs being
            learned, and thereby whether to include a new point or not. By default,
            we only consider the distance between points, using a threshold
            of 0.001 (determined in remove_close_points). Can also specify
            thresholds based on e.g. point-normal angle difference, or principal
            curvature magnitude difference.
        max_graph_size: Maximum size of a graph in meters. Any observations that fall
            out of this range will be discarded/used for building a new model. This
            constraints the size of models that an LM can learn and enforces learning
            models of sub-components of objects.
        max_nodes_per_graph: Maximum number of nodes in a graph. This will be k when
            picking the k-winner voxels to add their content into the graph used for
            matching.
        num_model_voxels_per_dim: Number of voxels per dimension in the model grid.
            This constraints the spatial resolution that the model can represent.
            max_graph_size/num_model_voxels_per_dim = how much space is lumped into one
            voxel. All locations that fall into the same voxel will be averaged and
            represented as one value. num_model_voxels_per_dim should not be too large
            since the memory requirements grow cubically with this number.
        gsg_class: The type of goal-state-generator to associate with the LM.
        gsg_args: Dictionary of configuration parameters for the GSG.

    Debugging Attributes:
        use_multithreading: Whether to calculate evidence updates for different
            objects in parallel using multithreading. This can be done since the
            updates to different objects are completely independent of each other. In
            general it is recommended to use this but it can be usefull to turn it off
            for debugging purposes.
    """

    def __init__(
        self,
        max_match_distance,
        tolerances,
        feature_weights,
        feature_evidence_increment=1,
        max_nneighbors=3,
        initial_possible_poses="informed",
        evidence_update_threshold="all",
        vote_evidence_threshold=0.8,
        past_weight=1,
        present_weight=1,
        vote_weight=1,
        object_evidence_threshold=1,
        x_percent_threshold=10,
        path_similarity_threshold=0.1,
        pose_similarity_threshold=0.35,
        required_symmetry_evidence=5,
        graph_delta_thresholds=None,
        max_graph_size=0.3,  # 30cm
        max_nodes_per_graph=2000,
        num_model_voxels_per_dim=50,  # -> voxel size = 6mm3 (0.006)
        use_multithreading=True,
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=None,
        *args,
        **kwargs,
    ):
        super(EvidenceGraphLM, self).__init__(
            initialize_base_modules=False, *args, **kwargs
        )
        # --- LM components ---
        self.graph_memory = EvidenceGraphMemory(
            graph_delta_thresholds=graph_delta_thresholds,
            max_nodes_per_graph=max_nodes_per_graph,
            max_graph_size=max_graph_size,
            num_model_voxels_per_dim=num_model_voxels_per_dim,
        )
        if gsg_args is None:
            gsg_args = dict()
        self.gsg = gsg_class(self, **gsg_args)
        self.gsg.reset()
        # --- Matching Params ---
        self.max_match_distance = max_match_distance
        self.tolerances = tolerances
        self.feature_evidence_increment = feature_evidence_increment
        self.max_nneighbors = max_nneighbors
        self.evidence_update_threshold = evidence_update_threshold
        self.vote_evidence_threshold = vote_evidence_threshold
        # ------ Weighting Params ------
        self.feature_weights = feature_weights
        self.past_weight = past_weight
        self.present_weight = present_weight
        self.vote_weight = vote_weight
        # --- Terminal Condition Params ---
        self.object_evidence_threshold = object_evidence_threshold
        self.x_percent_threshold = x_percent_threshold
        self.path_similarity_threshold = path_similarity_threshold
        self.pose_similarity_threshold = pose_similarity_threshold
        self.required_symmetry_evidence = required_symmetry_evidence
        # --- Model Params ---
        self.max_graph_size = max_graph_size
        # --- Debugging Params ---
        self.use_multithreading = use_multithreading

        # TODO make sure we always extract pose features and remove this
        self.tolerances = add_pose_features_to_tolerances(tolerances)
        # TODO: not ideal solution
        self.graph_memory.features_to_use = self.tolerances
        # Set feature weights to 1 if not otherwise specified
        self._fill_feature_weights_with_default(default=1)

        self.initial_possible_poses = get_initial_possible_poses(initial_possible_poses)
        self.use_features_for_matching = self._check_use_features_for_matching()

        # Dictionary with graph_ids as keys. For each graph we initialize a set of
        # hypotheses at the first step of an episode. Each hypothesis has an evidence
        # count associated with it which is stored here.
        # self.possible_locations and self.possible_poses have the same structure and
        # length as self.evidence and store the corresponding hypotheses.
        self.evidence = {}
        self.possible_locations = {}
        self.possible_poses = {}
        # Stores start and end indices of hypotheses in the above arrays for each graph
        # corresponding to each input channel. This is used to make sure the right
        # displacement is applied to the right hypotheses. Channel hypotheses are stored
        # contiguously so we can just specify ranges here.
        self.channel_hypothesis_mapping = {}

        self.current_mlh = {
            "graph_id": "no_observations_yet",
            "location": [0, 0, 0],
            "rotation": Rotation.from_euler("xyz", [0, 0, 0]),
            "scale": 1,
            "evidence": 0,
        }

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def reset(self):
        """Reset evidence count and other variables."""
        # Now here, as opposed to the displacement and feature-location LMs,
        # possible_matches is a list of IDs, not a dictionary with the object graphs.
        (
            self.possible_matches,
            self.possible_locations,
        ) = self.graph_memory.get_initial_hypotheses()

        if self.tolerances is not None:
            # TODO H: Differentiate between features from different input channels
            # TODO: could do this in the object model class
            self.graph_memory.initialize_feature_arrays()
        self.symmetry_evidence = 0
        self.last_possible_hypotheses = None

        self.current_mlh["graph_id"] = "no_observations_yet"
        self.current_mlh["location"] = [0, 0, 0]
        self.current_mlh["rotation"] = Rotation.from_euler("xyz", [0, 0, 0])
        self.current_mlh["scale"] = 1
        self.current_mlh["evidence"] = 0

    def receive_votes(self, vote_data):
        """Get evidence count votes and use to update own evidence counts.

        Weighted by distance to votes and their evidence.
        TODO: also take into account rotation vote

        vote_data contains:
            pos_location_votes: shape=(N, 3)
            pos_rotation_votes: shape=(N, 3, 3)
            pose_evidences: shape=(N,)

        """
        if (vote_data is not None) and (
            self.buffer.get_num_observations_on_object() > 0
        ):
            thread_list = []
            for graph_id in self.get_all_known_object_ids():
                if graph_id in vote_data.keys():
                    if self.use_multithreading:
                        t = threading.Thread(
                            target=self._update_evidence_with_vote,
                            args=(
                                vote_data[graph_id],
                                graph_id,
                            ),
                        )
                        thread_list.append(t)
                    else:  # This can be useful for debugging.
                        self._update_evidence_with_vote(
                            vote_data[graph_id],
                            graph_id,
                        )
            if self.use_multithreading:
                for thread in thread_list:
                    # start executing _update_evidence in each thread.
                    thread.start()
                for thread in thread_list:
                    # call this to prevent main thread from continuing in code
                    # before all evidences are updated.
                    thread.join()
            logging.debug("Updating possible matches after vote")
            self.possible_matches = self._threshold_possible_matches()
            self.current_mlh = self._calculate_most_likely_hypothesis()

            self._add_votes_to_buffer_stats(vote_data)

    def send_out_vote(self):
        """Send out hypotheses and the evidence for them.

        Votes are a dict and contain the following:
            pose_hypotheses: locations (V, 3) and rotations (V, 3, 3)
            pose_evidence: Evidence (V) for each location-rotation pair in the
                    pose hypotheses. Scaled into range [-1, 1] where 1 is the
                    hypothesis with the largest evidence in this LM and -1 the
                    one with the smallest evidence. When thresholded, pose_evidence
                    will be in range [self.vote_evidence_threshold, 1]
            sensed_pose_rel_body: sensed location and rotation of the input to this
                    LM. Rotation is represented by the pose vectors (point normal and
                    curvature directions) for the SMs. For input from LMs it is also
                    represented as 3 unit vectors, these are calculated from the
                    estimated rotation of the most likely object. This pose is used
                    to calculate the displacement between two voting LMs and to
                    translate the votes between their reference frames.
                    Shape=(4,3).
        Where V is the number of votes (V=number of hypotheses if not thresholded)
        If none of the hypotheses of an object are > vote_evidence_threshold, this
        object will not send out a vote.

        Returns:
            None or dict:
                possible_states: The possible states.
                sensed_pose_rel_body: The sensed pose relative to the body.
        """
        if (
            self.buffer.get_num_observations_on_object() < 1
            or not self.buffer.get_last_obs_processed()
        ):
            # We don't want the LM to vote if it hasn't gotten input yet (can happen
            # with multiple LMs if some start off the object) or if it didn't perform
            # an evidence update on this step (it didn't receive new input so it
            # doesn't have anything new to communicate).
            vote = None
        else:
            # Get pose of first sensor stored in buffer.
            sensed_pose = self.buffer.get_current_pose(input_channel="first")

            possible_states = dict()
            evidences = get_scaled_evidences(self.get_all_evidences())
            for graph_id in evidences.keys():
                interesting_hyp = np.where(
                    evidences[graph_id] > self.vote_evidence_threshold
                )
                if len(interesting_hyp[0]) > 0:
                    possible_states[graph_id] = []
                    for hyp_id in interesting_hyp[0]:
                        vote = State(
                            location=self.possible_locations[graph_id][
                                hyp_id
                            ],  # location rel. body
                            morphological_features={
                                # Pose vectors are columns of the rotation matrix
                                "pose_vectors": self.possible_poses[graph_id][hyp_id].T,
                                "pose_fully_defined": True,
                            },
                            # No feature when voting.
                            non_morphological_features=None,
                            confidence=evidences[graph_id][hyp_id],
                            use_state=True,
                            sender_id=self.learning_module_id,
                            sender_type="LM",
                        )
                        possible_states[graph_id].append(vote)

            vote = {
                "possible_states": possible_states,
                "sensed_pose_rel_body": sensed_pose,
            }
        return vote

    def get_output(self):
        """Return the most likely hypothesis in same format as LM input.

        The input to an LM at the moment is a dict of features at a location. The
        output therefor has the same format to keep the messaging protocol
        consistent and make it easy to stack multiple LMs on top of each other.

        If the evidence for mlh is < object_evidence_threshold,
        interesting_features == False
        """
        mlh = self.get_current_mlh()
        pose_features = self._object_pose_to_features(mlh["rotation"].inv())
        object_id_features = self._object_id_to_features(mlh["graph_id"])
        # Pass object ID to next LM if:
        #       1) The last input it received was on_object (+getting SM input
        #           check will make sure that we are also currently on object)
        #           NOTE: May want to relax this check but still need a motor input
        #       2) Its most likely hypothesis has an evidence >
        #           object_evidence_threshold
        use_state = bool(
            self.buffer.get_currently_on_object()
            and mlh["evidence"] > self.object_evidence_threshold
        )
        # TODO H: is this a good way to scale evidence to [0, 1]?
        confidence = (
            0
            if len(self.buffer) == 0
            else np.clip(mlh["evidence"] / len(self.buffer), 0, 1)
        )
        # TODO H1: update this to send detected object location
        # Use something like this + incorporate mlh location. -> while on same object,
        # this should not change, even when moving over the object. Would also have to
        # update mlh during exploration (just add displacenents).
        # Discuss this first before implementing. This would make higher level models
        # much simpler but also require some arbitrary object center and give less
        # resolution of where on a compositional object we are (in this lm). Would
        # also require update in terminal condition re. path_similarity_th.
        # object_loc_rel_body = (
        #     self.buffer.get_current_location(input_channel="first") - mlh["location"]
        # )
        hypothesized_state = State(
            # Same as input location from patch (rel body)
            # NOTE: Just for common format at the moment, movement information will be
            # taken from the sensor. For higher level LMs, we may want to transmit the
            # motor efference copy here.
            # TODO: get motor efference copy here. Need to refactor motor command
            # selection for this.
            location=self.buffer.get_current_location(
                input_channel="first"
            ),  # location rel. body
            morphological_features={
                "pose_vectors": pose_features,
                "pose_fully_defined": not self._enough_symmetry_evidence_accumulated(),
                "on_object": self.buffer.get_currently_on_object(),
            },
            non_morphological_features={
                "object_id": object_id_features,
                # TODO H: test if this makes sense to communicate
                "location_rel_model": mlh["location"],
            },
            confidence=confidence,
            use_state=use_state,
            sender_id=self.learning_module_id,
            sender_type="LM",
        )
        return hypothesized_state

    # ------------------ Getters & Setters ---------------------
    def set_detected_object(self, terminal_state):
        """Set the current graph ID.

        If we didn't recognize the object this will be new_object{n} where n is
        len(graph_memory) + 1. Otherwise it is the id of the graph that we recognized.
        If we timed out it is None and we will not update the graph memory.
        """
        self.terminal_state = terminal_state
        logging.debug(f"terminal state: {terminal_state}")
        if terminal_state is None:  # at beginning of episode
            graph_id = None
        elif (terminal_state == "no_match") or len(self.get_possible_matches()) == 0:
            if terminal_state == "time_out" or terminal_state == "pose_time_out":
                # If we have multiple LMs some of them might reach time out but with
                # no possible matches. In this case we don't want to add a new graph
                # to their memory.
                graph_id = None
            else:
                graph_id = "new_object" + str(len(self.graph_memory))

        elif terminal_state == "match":
            graph_id = self.get_possible_matches()[0]
        # If we are evaluating and reach a time out, we set the object to the
        # most likely hypothesis (if evidence for it is above object_evidence_threshold)
        elif self.mode == "eval" and (
            terminal_state == "time_out" or terminal_state == "pose_time_out"
        ):
            mlh = self.get_current_mlh()
            if "evidence" in mlh.keys() and (
                mlh["evidence"] > self.object_evidence_threshold
            ):
                # Use most likely hypothesis
                graph_id = mlh["graph_id"]
            else:
                graph_id = None
        else:
            graph_id = None
        self.detected_object = graph_id

    def get_unique_pose_if_available(self, object_id):
        """Get the most likely pose of an object if narrowed down.

        If there is not one unique possible pose or symmetry detected, return None

        Returns:
            The pose and scale if a unique pose is available, otherwise None.
        """
        possible_object_hypotheses_ids = self.get_possible_hypothesis_ids(object_id)
        # Only try to determine object pose if the evidence for it is high enough.
        if possible_object_hypotheses_ids is not None:
            mlh = self.get_current_mlh()
            # Check if all possible poses are similar
            pose_is_unique = self._check_for_unique_poses(
                object_id, possible_object_hypotheses_ids, mlh["rotation"]
            )
            # Check for symmetry
            symmetry_detected = self._check_for_symmetry(
                possible_object_hypotheses_ids,
                # Don't increment symmetry counter if LM didn't process observation
                increment_evidence=self.buffer.get_last_obs_processed(),
            )

            self.last_possible_hypotheses = possible_object_hypotheses_ids

            if pose_is_unique or symmetry_detected:
                r_inv = mlh["rotation"].inv()
                r_euler = mlh["rotation"].inv().as_euler("xyz", degrees=True)
                r_euler = np.round(r_euler, 3) % 360

                pose_and_scale = np.concatenate(
                    [mlh["location"], r_euler, [mlh["scale"]]]
                )
                logging.debug(f"(location, rotation, scale): {pose_and_scale}")
                # Set LM variables to detected object & pose
                self.detected_pose = pose_and_scale
                self.detected_rotation_r = mlh["rotation"]
                # Log stats to buffer
                lm_episode_stats = {
                    "detected_path": mlh["location"],
                    "detected_location_on_model": mlh["location"],
                    "detected_location_rel_body": self.buffer.get_current_location(
                        input_channel="first"
                    ),
                    "detected_rotation": r_euler,
                    "detected_rotation_quat": r_inv.as_quat(),
                    "detected_scale": 1,  # TODO: scale doesn't work yet
                }
                self.buffer.add_overall_stats(lm_episode_stats)
                if symmetry_detected:
                    symmetry_stats = {
                        "symmetric_rotations": np.array(self.possible_poses[object_id])[
                            self.last_possible_hypotheses
                        ],
                        "symmetric_locations": self.possible_locations[object_id][
                            self.last_possible_hypotheses
                        ],
                    }
                    self.buffer.add_overall_stats(symmetry_stats)
                return pose_and_scale
            else:
                logging.debug(f"object {object_id} detected but pose not resolved yet.")
                return None
        else:
            return None

    def get_current_mlh(self):
        """Return the current most likely hypothesis of the learning module.

        Returns:
            dict with keys: graph_id, location, rotation, scale, evidence
        """
        return self.current_mlh

    def get_mlh_for_object(self, object_id):
        """Get mlh for a specific object ID.

        Note:
            When trying to retrieve the MLH for the current most likely object
            and not any other object, it is better to use self.current_mlh

        Returns:
            The most likely hypothesis for the object ID.
        """
        return self._calculate_most_likely_hypothesis(object_id)

    def get_top_two_mlh_ids(self):
        """Retrieve the two most likely object IDs for this LM.

        Returns:
            The two most likely object IDs.
        """
        graph_ids, graph_evidences = self.get_evidence_for_each_graph()

        # Note the indices below will be ordered with the 2nd MLH appearing first, and
        # the 1st MLH appearing second.
        top_indices = np.argsort(graph_evidences)[-2:]

        if len(top_indices) > 1:
            top_id = graph_ids[top_indices[1]]
            second_id = graph_ids[top_indices[0]]
        else:
            top_id = graph_ids[top_indices[0]]
            second_id = top_id

        return top_id, second_id

    def get_top_two_pose_hypotheses_for_graph_id(self, graph_id):
        """Return top two hypotheses for a given graph_id."""
        mlh_for_graph = self._calculate_most_likely_hypothesis(graph_id)
        second_mlh_id = np.argsort(self.evidence[graph_id])[-2]
        second_mlh = self._get_mlh_dict_from_id(graph_id, second_mlh_id)
        return mlh_for_graph, second_mlh

    def get_possible_matches(self):
        """Return graph ids with significantly higher evidence than median."""
        return self.possible_matches

    def get_possible_poses(self, as_euler=True):
        """Return possible poses for each object (for logging).

        Here this list doesn't get narrowed down.
        This is not really used for evidence matching since we threshold in other
        places.
        """
        poses = self.possible_poses.copy()
        if as_euler:
            all_poses = dict()
            for obj in poses.keys():
                euler_poses = []
                for pose in poses[obj]:
                    scipy_pose = Rotation.from_matrix(pose)
                    euler_pose = np.round(
                        scipy_pose.inv().as_euler("xyz", degrees=True), 5
                    )
                    euler_poses.append(euler_pose)
                all_poses[obj] = euler_poses
        else:
            all_poses = poses
        return all_poses

    def get_possible_hypothesis_ids(self, object_id):
        max_obj_evidence = np.max(self.evidence[object_id])
        # TODO: Try out different ways to adapt object_evidence_threshold to number of
        # steps taken so far and number of objects in memory
        if max_obj_evidence > self.object_evidence_threshold:
            x_percent_of_max = max_obj_evidence / 100 * self.x_percent_threshold
            # Get all pose IDs that have an evidence in the top n%
            possible_object_hypotheses_ids = np.where(
                self.evidence[object_id] > max_obj_evidence - x_percent_of_max
            )[0]
            logging.debug(
                f"possible hpids: {possible_object_hypotheses_ids} for {object_id}"
            )
            logging.debug(f"hpid evidence is > {max_obj_evidence} - {x_percent_of_max}")
            return possible_object_hypotheses_ids

    def get_evidence_for_each_graph(self):
        """Return maximum evidence count for a pose on each graph."""
        graph_ids = self.get_all_known_object_ids()
        if graph_ids[0] not in self.evidence.keys():
            return ["patch_off_object"], [0]
        graph_evidences = []
        for graph_id in graph_ids:
            graph_evidences.append(np.max(self.evidence[graph_id]))
        return graph_ids, np.array(graph_evidences)

    def get_all_evidences(self):
        """Return evidence for each pose on each graph (pointer)."""
        return self.evidence

    # ------------------ Logging & Saving ----------------------
    def collect_stats_to_save(self):
        """Get all stats that this LM should store in the buffer for logging.

        Returns:
            The stats dictionary.
        """
        stats = {
            "possible_matches": self.get_possible_matches(),
            "current_mlh": self.get_current_mlh(),
        }
        if self.has_detailed_logger:
            # Save possible poses once since they don't change during episode
            get_rotations = False
            if "possible_rotations" not in self.buffer.stats.keys():
                get_rotations = True

            stats = self._add_detailed_stats(stats, get_rotations)
        return stats

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _get_initial_hypothesis_space(self, features, graph_id, input_channel):
        if self.initial_possible_poses is None:
            # Get initial poses for all locations informed by pose features
            (
                initial_possible_channel_locations,
                initial_possible_channel_rotations,
            ) = self._get_all_informed_possible_poses(graph_id, features, input_channel)
        else:
            initial_possible_channel_locations = []
            initial_possible_channel_rotations = []
            all_channel_locations = self.graph_memory.get_locations_in_graph(
                graph_id, input_channel
            )
            # Initialize fixed possible poses (without using pose features)
            for node_id in range(len(all_channel_locations)):
                for rotation in self.initial_possible_poses:
                    initial_possible_channel_locations.append(
                        all_channel_locations[node_id]
                    )
                    initial_possible_channel_rotations.append(rotation.as_matrix())

            initial_possible_channel_rotations = np.array(
                initial_possible_channel_rotations
            )
        # There will always be two feature weights (point normal and curvature
        # direction). If there are no more weight we are not using features for
        # matching and skip this step. Doing matching with only morphology can
        # currently be achieved in two ways. Either we don't specify tolerances
        # and feature_weights or we set the global feature_evidence_increment to 0.
        if self.use_features_for_matching[input_channel]:
            # Get real valued features match for each node
            node_feature_evidence = self._calculate_feature_evidence_for_all_nodes(
                features, input_channel, graph_id
            )
            # stack node_feature_evidence to match possible poses
            nwmf_stacked = []
            for _ in range(
                len(initial_possible_channel_rotations) // len(node_feature_evidence)
            ):
                nwmf_stacked.extend(node_feature_evidence)
            # add evidence if features match
            evidence = np.array(nwmf_stacked) * self.feature_evidence_increment
        else:
            evidence = np.zeros((initial_possible_channel_rotations.shape[0]))
        return (
            initial_possible_channel_locations,
            initial_possible_channel_rotations,
            evidence,
        )

    def _add_hypotheses_to_hpspace(
        self,
        graph_id,
        input_channel,
        new_loc_hypotheses,
        new_rot_hypotheses,
        new_evidence,
    ):
        """Add new hypotheses to hypothesis space."""
        # Add current mean evidence to give the new hypotheses a fighting
        # chance. TODO H: Test mean vs. median here.
        current_mean_evidence = np.mean(self.evidence[graph_id])
        new_evidence = new_evidence + current_mean_evidence
        # Add new hypotheses to hypothesis space
        self.possible_locations[graph_id] = np.vstack(
            [
                self.possible_locations[graph_id],
                new_loc_hypotheses,
            ]
        )
        self.possible_poses[graph_id] = np.vstack(
            [
                self.possible_poses[graph_id],
                new_rot_hypotheses,
            ]
        )
        self.evidence[graph_id] = np.hstack([self.evidence[graph_id], new_evidence])
        # Update channel hypothesis mapping
        old_num_hypotheses = self.channel_hypothesis_mapping[graph_id]["num_hypotheses"]
        new_num_hypotheses = old_num_hypotheses + len(new_loc_hypotheses)
        self.channel_hypothesis_mapping[graph_id][input_channel] = [
            old_num_hypotheses,
            new_num_hypotheses,
        ]
        self.channel_hypothesis_mapping[graph_id]["num_hypotheses"] = new_num_hypotheses

    def _update_possible_matches(self, query):
        """Update evidence for each hypothesis instead of removing them."""
        thread_list = []
        for graph_id in self.get_all_known_object_ids():
            if self.use_multithreading:
                # assign separate thread on same CPU to each objects update.
                # Since the updates of different objects are independent of
                # each other we can do this.
                t = threading.Thread(
                    target=self._update_evidence,
                    args=(query[0], query[1], graph_id),
                )
                thread_list.append(t)
            else:  # This can be useful for debugging.
                self._update_evidence(query[0], query[1], graph_id)
        if self.use_multithreading:
            # TODO: deal with keyboard interrupt
            for thread in thread_list:
                # start executing _update_evidence in each thread.
                thread.start()
            for thread in thread_list:
                # call this to prevent main thread from continuing in code
                # before all evidences are updated.
                thread.join()
        # NOTE: would not need to do this if we are still voting
        # Call this update in the step method?
        self.possible_matches = self._threshold_possible_matches()
        self.current_mlh = self._calculate_most_likely_hypothesis()

    def _update_evidence(self, features, displacements, graph_id):
        """Update evidence for poses of graph_id.

        replaces _update_matches_using_features

        - start with 0 evidence -> flat prior
        - post-features & displacement: +/- evidence
        - features: + evidence
        Evidence is weighted by distance of hypothesis to point in model.
        ----------not included in this function------
        - votes: add average of nearby incoming evidence votes (could also be weighted
          by distance)
        ----------added later on perhaps?-----
        - displacement recognition (stored edges in graph as ppf): + evidence for
          locations

        if first step (not moved yet):
            - get initial hypothesis space using observed pose features
            - initialize evidence for hypotheses using first observed features
        else:
            - update evidence for hypotheses using all observed features
            and displacement

        Raises:
            ValueError: If no input channels are found to initializing hypotheses
        """
        start_time = time.time()
        all_input_channels = list(features.keys())
        input_channels_to_use = []
        for input_channel in all_input_channels:
            if input_channel in self.get_input_channels_in_graph(graph_id):
                # NOTE: We might also want to check the confidence in the input channel
                # features. This information is currently not available here. Once we
                # pull the observation class into the LM we could add this (TODO S).
                input_channels_to_use.append(input_channel)
        # Before moving we initialize the hypothesis space:
        if displacements is None:
            if len(input_channels_to_use) == 0:
                # QUESTION: Do we just want to continue until we get input?
                raise ValueError(
                    "No input channels found to initializing hypotheses. Make sure"
                    " there is at least one channel that is also stored in the graph."
                )
            # This is the first observation before we moved -> check where in the
            # graph the feature can be found and initialize poses & evidence
            initial_possible_locations = []
            initial_possible_rotations = []
            initial_evidence = []
            self.channel_hypothesis_mapping[graph_id] = dict()
            num_hypotheses = 0
            for input_channel in input_channels_to_use:
                (
                    initial_possible_channel_locations,
                    initial_possible_channel_rotations,
                    channel_evidence,
                ) = self._get_initial_hypothesis_space(
                    features, graph_id, input_channel
                )
                initial_possible_locations.append(initial_possible_channel_locations)
                initial_possible_rotations.append(initial_possible_channel_rotations)
                initial_evidence.append(channel_evidence)
                self.channel_hypothesis_mapping[graph_id][input_channel] = [
                    num_hypotheses,
                    num_hypotheses + len(initial_possible_channel_locations),
                ]
                num_hypotheses += len(initial_possible_channel_locations)
            self.channel_hypothesis_mapping[graph_id]["num_hypotheses"] = num_hypotheses
            self.possible_locations[graph_id] = np.concatenate(
                initial_possible_locations, axis=0
            )
            self.possible_poses[graph_id] = np.concatenate(
                initial_possible_rotations, axis=0
            )
            self.evidence[graph_id] = (
                np.concatenate(initial_evidence, axis=0) * self.present_weight
            )
            logging.debug(
                f"\nhypothesis space for {graph_id}: {self.evidence[graph_id].shape[0]}"
            )
            assert (
                self.evidence[graph_id].shape[0]
                == self.possible_locations[graph_id].shape[0]
            )
        # ---------------------------------------------------------------------------
        # Use displacement and new sensed features to update evidence for hypotheses.
        else:
            if len(input_channels_to_use) == 0:
                logging.info(
                    f"No input channels observed for {graph_id} that are stored in . "
                    "the model. Not updating evidence."
                )
            for input_channel in input_channels_to_use:
                # If channel features are observed for the first time, initialize
                # hypotheses for them.
                if (
                    input_channel
                    not in self.channel_hypothesis_mapping[graph_id].keys()
                ):
                    # TODO H: When initializing a hypothesis for a channel later on,
                    # include most likely existing hypothesis from other channels?
                    (
                        initial_possible_channel_locations,
                        initial_possible_channel_rotations,
                        channel_evidence,
                    ) = self._get_initial_hypothesis_space(
                        features, graph_id, input_channel
                    )

                    self._add_hypotheses_to_hpspace(
                        graph_id=graph_id,
                        input_channel=input_channel,
                        new_loc_hypotheses=initial_possible_channel_locations,
                        new_rot_hypotheses=initial_possible_channel_rotations,
                        new_evidence=channel_evidence,
                    )

                else:
                    # Get the observed displacement for this channel
                    displacement = displacements[input_channel]
                    # Get the IDs in hypothesis space for this channel
                    channel_start, channel_end = self.channel_hypothesis_mapping[
                        graph_id
                    ][input_channel]
                    # Have to do this for all hypotheses so we don't loose the path
                    # information
                    rotated_displacements = self.possible_poses[graph_id][
                        channel_start:channel_end
                    ].dot(displacement)
                    search_locations = (
                        self.possible_locations[graph_id][channel_start:channel_end]
                        + rotated_displacements
                    )
                    # Threshold hypotheses that we update by evidence for them
                    current_evidence_update_threshold = (
                        self._get_evidence_update_threshold(graph_id)
                    )
                    # Get indices of hypotheses with evidence > threshold
                    hyp_ids_to_test = np.where(
                        self.evidence[graph_id][channel_start:channel_end]
                        >= current_evidence_update_threshold
                    )[0]
                    num_hypotheses_to_test = hyp_ids_to_test.shape[0]
                    if num_hypotheses_to_test > 0:
                        logging.info(
                            f"Testing {num_hypotheses_to_test} out of "
                            f"{self.evidence[graph_id].shape[0]} hypotheses for "
                            f"{graph_id} "
                            f"(evidence > {current_evidence_update_threshold})"
                        )
                        search_locations_to_test = search_locations[hyp_ids_to_test]
                        # Get evidence update for all hypotheses with evidence > current
                        # _evidence_update_threshold
                        new_evidence = self._calculate_evidence_for_new_locations(
                            graph_id,
                            input_channel,
                            search_locations_to_test,
                            features,
                            hyp_ids_to_test,
                        )
                        min_update = np.clip(np.min(new_evidence), 0, np.inf)
                        # Alternatives (no update to other Hs or adding avg) left in
                        # here in case we want to revert back to those.
                        # avg_update = np.mean(new_evidence)
                        # evidence_to_add = np.zeros_like(self.evidence[graph_id])
                        evidence_to_add = (
                            np.ones_like(
                                self.evidence[graph_id][channel_start:channel_end]
                            )
                            * min_update
                        )
                        evidence_to_add[hyp_ids_to_test] = new_evidence
                        # If past and present weight add up to 1, equivalent to
                        # np.average and evidence will be bound to [-1, 2]. Otherwise it
                        # keeps growing.
                        self.evidence[graph_id][channel_start:channel_end] = (
                            self.evidence[graph_id][channel_start:channel_end]
                            * self.past_weight
                            + evidence_to_add * self.present_weight
                        )
                    self.possible_locations[graph_id][channel_start:channel_end] = (
                        search_locations
                    )
        end_time = time.time()
        assert not np.isnan(np.max(self.evidence[graph_id])), "evidence contains NaN."
        logging.debug(
            f"evidence update for {graph_id} took "
            f"{np.round(end_time - start_time,2)} seconds."
            f" New max evidence: {np.round(np.max(self.evidence[graph_id]),3)}"
        )

    def _update_evidence_with_vote(self, state_votes, graph_id):
        """Use incoming votes to update all hypotheses."""
        # Extract information from list of State classes into np.arrays for efficient
        # matrix operations and KDTree search.
        graph_location_vote = np.zeros((len(state_votes), 3))
        vote_evidences = np.zeros(len(state_votes))
        for n, vote in enumerate(state_votes):
            graph_location_vote[n] = vote.location
            vote_evidences[n] = vote.confidence

        vote_location_tree = KDTree(
            graph_location_vote,
            leafsize=40,
        )
        vote_nn = 3  # TODO: Make this a parameter?
        if graph_location_vote.shape[0] < vote_nn:
            vote_nn = graph_location_vote.shape[0]
        # Get max_nneighbors closest nodes and their distances
        (radius_node_dists, radius_node_ids) = vote_location_tree.query(
            self.possible_locations[graph_id],
            k=vote_nn,
            p=2,
            workers=1,
        )
        if vote_nn == 1:
            radius_node_dists = np.expand_dims(radius_node_dists, axis=1)
            radius_node_ids = np.expand_dims(radius_node_ids, axis=1)
        radius_evidences = vote_evidences[radius_node_ids]
        # Check that nearest node are in the radius
        node_distance_weights = self._get_node_distance_weights(radius_node_dists)
        too_far_away = node_distance_weights <= 0
        # Mask the votes which are too far away
        all_radius_evidence = np.ma.array(radius_evidences, mask=too_far_away)
        # Get the highest vote in the radius. Currently unweighted but using
        # np.ma.average and the node_distance_weights also works reasonably well.
        distance_weighted_vote_evidence = np.ma.max(
            all_radius_evidence,
            # weights=node_distance_weights,
            axis=1,
        )

        if self.past_weight + self.present_weight == 1:
            # Take the average to keep evidence in range
            self.evidence[graph_id] = np.ma.average(
                [
                    self.evidence[graph_id],
                    distance_weighted_vote_evidence,
                ],
                weights=[1, self.vote_weight],
                axis=0,
            )
        else:
            # Add to evidence count if the evidence can grow infinitely. Taking the
            # average would drag down the evidence otherwise.
            self.evidence[graph_id] = np.ma.sum(
                [
                    self.evidence[graph_id],
                    distance_weighted_vote_evidence * self.vote_weight,
                ],
                axis=0,
            )

    def _calculate_evidence_for_new_locations(
        self,
        graph_id,
        input_channel,
        search_locations,
        features,
        hyp_ids_to_test,
    ):
        """Use search locations, sensed features and graph model to calculate evidence.

        First, the search locations are used to find the nearest nodes in the graph
        model. Then we calculate the error between the stored pose features and the
        sensed ones. Additionally we look at whether the non-pose features match at the
        neigboring nodes. Everything is weighted by the nodes distance from the search
        location.
        If there are no nodes in the search radius (max_match_distance), evidence = -1.

        We do this for every incoming input channel and its features if they are stored
        in the graph and take the average over the evidence from all input channels.

        Returns:
            The location evidence.
        """
        logging.debug(
            f"Calculating evidence for {graph_id} using input from " f"{input_channel}"
        )

        pose_transformed_features = rotate_pose_dependent_features(
            features[input_channel],
            self.possible_poses[graph_id][hyp_ids_to_test],
        )
        # Get max_nneighbors nearest nodes to search locations.
        nearest_node_ids = self.get_graph(
            graph_id, input_channel
        ).find_nearest_neighbors(
            search_locations,
            num_neighbors=self.max_nneighbors,
        )
        if self.max_nneighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

        nearest_node_locs = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        )[nearest_node_ids]
        max_abs_curvature = get_relevant_curvature(features[input_channel])
        custom_nearest_node_dists = get_custom_distances(
            nearest_node_locs,
            search_locations,
            pose_transformed_features["pose_vectors"][:, 0],
            max_abs_curvature,
        )
        # shape=(H, K)
        node_distance_weights = self._get_node_distance_weights(
            custom_nearest_node_dists
        )
        # Get IDs where custom_nearest_node_dists > max_match_distance
        mask = node_distance_weights <= 0

        new_pos_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            nearest_node_ids,
            feature_keys=["pose_vectors", "pose_fully_defined"],
        )
        # Calculate the pose error for each hypothesis
        # shape=(H, K)
        radius_evidence = self._get_pose_evidence_matrix(
            pose_transformed_features,
            new_pos_features,
            input_channel,
            node_distance_weights,
        )
        # Set the evidences which are too far away to -1
        radius_evidence[mask] = -1
        # If a node is too far away, weight the negative evidence fully (*1). This
        # only comes into play if there are no nearby nodes in the radius, then we
        # want an evidence of -1 for this hypothesis.
        # NOTE: Currently we don't weight the evidence by distance so this doesn't
        # matter.
        node_distance_weights[mask] = 1

        # If no feature weights are provided besides the ones for point_normal
        # and curvature_directions we don't need to calculate feature evidence.
        if self.use_features_for_matching[input_channel]:
            # add evidence if features match
            node_feature_evidence = self._calculate_feature_evidence_for_all_nodes(
                features, input_channel, graph_id
            )
            hypothesis_radius_feature_evidence = node_feature_evidence[nearest_node_ids]
            # Set feature evidence of nearest neighbors that are too far away to 0
            hypothesis_radius_feature_evidence[mask] = 0
            # Take the maximum feature evidence out of the nearest neighbors in the
            # search radius and weighted by its distance to the search location.
            # Evidence will be in [0, 1] and is only 1 if all features match
            # perfectly and the node is at the search location.
            radius_evidence = (
                radius_evidence
                + hypothesis_radius_feature_evidence * self.feature_evidence_increment
            )
        # We take the maximum to be better able to deal with parts of the model where
        # features change quickly and we may have noisy location information. This way
        # we check if we can find a good match of pose features within the search
        # radius. It doesn't matter if there are also points stored nearby in the model
        # that are not a good match.
        # Removing the comment weights the evidence by the nodes distance from the
        # search location. However, epirically this did not seem to help.
        # shape=(H,)
        location_evidence = np.max(
            radius_evidence,  # * node_distance_weights,
            axis=1,
        )
        return location_evidence

    def _get_pose_evidence_matrix(
        self,
        query_features,
        node_features,
        input_channel,
        node_distance_weights,
    ):
        """Get angle mismatch error of the three pose features for multiple points.

        Args:
            query_features: Observed features.
            node_features: Features at nodes that are being tested.
            input_channel: Input channel for which we want to calculate the
                pose evidence. This are all input channels that are received at the
                current time step and are also stored in the graph.
            node_distance_weights: Weights for each nodes error (determined by
                distance to the search location). Currently not used, except for shape.

        Returns:
            The sum of angle evidence weighted by weights. In range [-1, 1].
        """
        # TODO S: simplify by looping over pose vectors
        evidences_shape = node_distance_weights.shape[:2]
        pose_evidence_weighted = np.zeros((evidences_shape))
        # TODO H: at higher level LMs we may want to look at all pose vectors.
        # Currently we skip the third since the second curv dir is always 90 degree
        # from the first.
        # Get angles between three pose features
        pn_error = get_angles_for_all_hypotheses(
            # shape of node_features[input_channel]["pose_vectors"]: (nH, knn, 9)
            node_features["pose_vectors"][:, :, :3],
            query_features["pose_vectors"][:, 0],  # shape (nH, 3)
        )
        # Divide error by 2 so it is in range [0, pi/2]
        # Apply sin -> [0, 1]. Subtract 0.5 -> [-0.5, 0.5]
        # Negate the error to get evidence (lower error is higher evidence)
        pn_evidence = -(np.sin(pn_error / 2) - 0.5)
        pn_weight = self.feature_weights[input_channel]["pose_vectors"][0]
        # If curvatures are same the directions are meaningless
        #  -> set curvature angle error to zero.
        if not query_features["pose_fully_defined"]:
            cd1_weight = 0
            # Only calculate curv dir angle if sensed curv dirs are meaningful
            cd1_evidence = np.zeros(pn_error.shape)
        else:
            cd1_weight = self.feature_weights[input_channel]["pose_vectors"][1]
            # Also check if curv dirs stored at node are meaningful
            use_cd = np.array(
                node_features["pose_fully_defined"][:, :, 0],
                dtype=bool,
            )
            cd1_angle = get_angles_for_all_hypotheses(
                node_features["pose_vectors"][:, :, 3:6],
                query_features["pose_vectors"][:, 1],
            )
            # Since curvature directions could be rotated 180 degrees we define the
            # error to be largest when the angle is pi/2 (90 deg) and angles 0 and
            # pi are equal. This means the angle error will be between 0 and pi/2.
            cd1_error = np.pi / 2 - np.abs(cd1_angle - np.pi / 2)
            # We then apply the same operations as on pn error to get cd1_evidence
            # in range [-0.5, 0.5]
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            # nodes where pc1==pc2 receive no cd evidence but twice the pn evidence
            # -> overall evidence can be in range [-1, 1]
            cd1_evidence = cd1_evidence * use_cd
            pn_evidence[np.logical_not(use_cd)] * 2
        # weight angle errors by feature weights
        # if sensed pc1==pc2 cd1_weight==0 and overall evidence is in [-0.5, 0.5]
        # otherwise it is in [-1, 1].
        pose_evidence_weighted += pn_evidence * pn_weight + cd1_evidence * cd1_weight
        return pose_evidence_weighted

    def _calculate_feature_evidence_for_all_nodes(
        self, query_features, input_channel, graph_id
    ):
        """Calculate the feature evidence for all nodes stored in a graph.

        Evidence for each feature depends on the difference between observed and stored
        features, feature weights, and distance weights.

        Evidence is a float between 0 and 1. An evidence of 1 is a perfect match, the
        larger the difference between observed and sensed features, the close to 0 goes
        the evidence. Evidence is 0 if the difference is >= the tolerance for this
        feature.

        If a node does not store a given feature, evidence will be nan.

        input_channel indicates where the sensed features are coming from and thereby
        tells this function to which features in the graph they need to be compared.

        Returns:
            The feature evidence for all nodes.
        """
        feature_array = self.graph_memory.get_feature_array(graph_id)
        feature_order = self.graph_memory.get_feature_order(graph_id)
        # generate the lists of features, tolerances, and whether features are circular
        shape_to_use = feature_array[input_channel].shape[1]
        feature_order = feature_order[input_channel]
        tolerance_list = np.zeros(shape_to_use) * np.nan
        feature_weight_list = np.zeros(shape_to_use) * np.nan
        feature_list = np.zeros(shape_to_use) * np.nan
        circular_var = np.zeros(shape_to_use, dtype=bool)
        start_idx = 0
        query_features = query_features[input_channel]
        for feature in feature_order:
            if feature in [
                "pose_vectors",
                "pose_fully_defined",
            ]:
                continue
            if hasattr(query_features[feature], "__len__"):
                feature_length = len(query_features[feature])
            else:
                feature_length = 1
            end_idx = start_idx + feature_length
            feature_list[start_idx:end_idx] = query_features[feature]
            tolerance_list[start_idx:end_idx] = self.tolerances[input_channel][feature]
            feature_weight_list[start_idx:end_idx] = self.feature_weights[
                input_channel
            ][feature]
            circular_var[start_idx:end_idx] = (
                [True, False, False] if feature == "hsv" else False
            )
            circ_range = 1
            start_idx = end_idx

        feature_differences = np.zeros_like(feature_array[input_channel])
        feature_differences[:, ~circular_var] = np.abs(
            feature_array[input_channel][:, ~circular_var] - feature_list[~circular_var]
        )
        cnode_fs = feature_array[input_channel][:, circular_var]
        cquery_fs = feature_list[circular_var]
        feature_differences[:, circular_var] = np.min(
            [
                np.abs(circ_range + cnode_fs - cquery_fs),
                np.abs(cnode_fs - cquery_fs),
                np.abs(cnode_fs - (cquery_fs + circ_range)),
            ],
            axis=0,
        )
        # any difference < tolerance should be positive evidence
        # any difference >= tolerance should be 0 evidence
        feature_evidence = np.clip(tolerance_list - feature_differences, 0, np.inf)
        # normalize evidence to be in [0, 1]
        feature_evidence = feature_evidence / tolerance_list
        weighted_feature_evidence = np.average(
            feature_evidence, weights=feature_weight_list, axis=1
        )
        return weighted_feature_evidence

    def _check_for_unique_poses(
        self,
        graph_id,
        possible_object_hypotheses_ids,
        most_likely_r,
    ):
        """Check if we have the pose of an object narrowed down.

        This method checks two things:
        - all possible locations are in a radius < path_similarity_threshold
        - all possible rotations have an angle < pose_similarity_threshold between
          each other
        If both are True, return True, else False

        Returns:
            Whether the pose is unique.
        """
        possible_locations = np.array(
            self.possible_locations[graph_id][possible_object_hypotheses_ids]
        )
        logging.debug(f"{possible_locations.shape[0]} possible locations")

        center_location = np.mean(possible_locations, axis=0)
        distances_to_center = np.linalg.norm(
            possible_locations - center_location, axis=1
        )
        location_unique = np.max(distances_to_center) < self.path_similarity_threshold
        if location_unique:
            logging.info(
                "all possible locations are in radius "
                f"{self.path_similarity_threshold} of {center_location}"
            )

        possible_rotations = np.array(self.possible_poses[graph_id])[
            possible_object_hypotheses_ids
        ]
        logging.debug(f"{possible_rotations.shape[0]} possible rotations")

        # Compute the difference between each rotation matrix in the list of possible
        # rotations and the most likely rotation in radians.
        trace = np.trace(
            most_likely_r.as_matrix().T @ possible_rotations, axis1=1, axis2=2
        )
        differences = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        # Check if none of them differ by more than pose_similarity_threshold
        rotation_unique = (
            np.max(np.nan_to_num(differences)) <= self.pose_similarity_threshold
        )

        pose_is_unique = location_unique and rotation_unique
        return pose_is_unique

    def _check_for_symmetry(self, possible_object_hypotheses_ids, increment_evidence):
        """Check whether the most likely hypotheses stayed the same over the past steps.

        Since the definition of possible_object_hypotheses is a bit murky and depends
        on how we set an evidence threshold we check the set overlap here and see if at
        least 90% of the current hypotheses were also possible on the last step. I'm
        not sure if this is the best way to check for symmetry...

        Args:
            possible_object_hypotheses_ids: List of IDs of all possible hypotheses.
            increment_evidence: Whether to increment symmetry evidence or not. We
                may want this to be False for example if we did not receive a new
                observation.

        Returns:
            Whether symmetry was detected.
        """
        if self.last_possible_hypotheses is None:
            return False  # need more steps to meet symmetry condition
        logging.debug(
            f"\n\nchecking for symmetry for hp ids {possible_object_hypotheses_ids}"
            f" with last ids {self.last_possible_hypotheses}"
        )
        if increment_evidence:
            previous_hyps = set(possible_object_hypotheses_ids)
            current_hyps = set(self.last_possible_hypotheses)
            hypothesis_overlap = previous_hyps.intersection(current_hyps)
            if len(hypothesis_overlap) / len(current_hyps) > 0.9:
                # at least 90% of current possible ids were also in previous ids
                logging.info("added symmetry evidence")
                self.symmetry_evidence += 1
            else:  # has to be consequtive
                self.symmetry_evidence = 0

        if self._enough_symmetry_evidence_accumulated():
            logging.info(
                f"Symmetry detected for hypotheses {possible_object_hypotheses_ids}"
            )
            return True
        else:
            return False

    def _enough_symmetry_evidence_accumulated(self):
        """Check if enough evidence for symmetry has been accumulated.

        Note:
            Code duplication from FeatureGraphMemory

        Returns:
            Whether enough evidence for symmetry has been accumulated.
        """
        return self.symmetry_evidence >= self.required_symmetry_evidence

    def _object_pose_to_features(self, pose):
        """Turn object rotation into pose feature like vectors.

        pose is a rotation matrix. We multiply rf spanning unit vectors with rotation.

        Returns:
            The pose features.
        """
        if pose is None:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            # Equivalent to applying the pose rotation to the rf spanning unit vectors
            # -> pose.as_matrix().dot(pose_vectors.T).T
            return pose.as_matrix().T

    def _object_id_to_features(self, object_id):
        """Turn object ID into features that express object similarity.

        Returns:
            The object ID features.
        """
        # TODO H: Make this based on object similarity
        # For now just taking sum of character ids in object name
        id_feature = sum([ord(i) for i in object_id])
        return id_feature

    # ------------------------ Helper --------------------------
    def _check_use_features_for_matching(self):
        use_features = dict()
        for input_channel in self.tolerances.keys():
            if input_channel not in self.feature_weights.keys():
                use_features[input_channel] = False
            elif self.feature_evidence_increment <= 0:
                use_features[input_channel] = False
            else:
                feature_weights_provided = (
                    len(self.feature_weights[input_channel].keys()) > 2
                )
                use_features[input_channel] = feature_weights_provided
        return use_features

    def _fill_feature_weights_with_default(self, default):
        for input_channel in self.tolerances.keys():
            if input_channel not in self.feature_weights.keys():
                self.feature_weights[input_channel] = dict()
            for key in self.tolerances[input_channel].keys():
                if key not in self.feature_weights[input_channel].keys():
                    if hasattr(self.tolerances[input_channel][key], "shape"):
                        shape = self.tolerances[input_channel][key].shape
                    elif hasattr(self.tolerances[input_channel][key], "__len__"):
                        shape = len(self.tolerances[input_channel][key])
                    else:
                        shape = 1
                    default_weights = np.ones(shape) * default
                    logging.debug(
                        f"adding {key} to feature_weights with value {default_weights}"
                    )
                    self.feature_weights[input_channel][key] = default_weights

    def _get_all_informed_possible_poses(
        self, graph_id, sensed_features, input_channel
    ):
        """Initialize hypotheses on possible rotations for each location.

        Similar to _get_informed_possible_poses but doesn't require looping over nodes

        For this we use the point normal and curvature directions and check how
        they would have to be rotated to match between sensed and stored vectors
        at each node. If principal curvature is similar in both directions, the
        direction vectors cannot inform this and we have to uniformly sample multiple
        possible rotations along this plane.

        Note:
            In general this initialization of hypotheses determines how well the
            matching later on does and if an object and pose can be recognized. We
            should think about whether this is the most robust way to initialize
            hypotheses.

        Returns:
            The possible locations and rotations.
        """
        all_possible_locations = np.zeros((1, 3))
        all_possible_rotations = np.zeros((1, 3, 3))

        logging.debug("Determining possible poses using input from " f"{input_channel}")
        node_directions = self.graph_memory.get_rotation_features_at_all_nodes(
            graph_id, input_channel
        )
        sensed_directions = sensed_features[input_channel]["pose_vectors"]
        # Check if PCs in patch are similar -> need to sample more directions
        if (
            "pose_fully_defined" in sensed_features[input_channel].keys()
            and not sensed_features[input_channel]["pose_fully_defined"]
        ):
            sample_more_directions = True
        else:
            sample_more_directions = False

        if not sample_more_directions:
            # 2 possibilities since the curvature directions may be flipped
            possible_s_d = [
                sensed_directions.copy(),
                sensed_directions.copy(),
            ]
            possible_s_d[1][1:] = possible_s_d[1][1:] * -1
        else:
            # TODO: whats a reasonable number here?
            # Maybe just samle n poses regardless of if pc1==pc2 and increase
            # evidence in the cases where we are more sure?
            # Maybe keep moving until pc1!= pc2 and then start matching?
            possible_s_d = get_more_directions_in_plane(sensed_directions, 8)

        for s_d in possible_s_d:
            # Since we have orthonormal vectors and know their correspondence we can
            # directly calculate the rotation instead of using the Kabsch esimate
            # used in Rotation.align_vectors
            r = align_multiple_orthonormal_vectors(node_directions, s_d, as_scipy=False)
            all_possible_locations = np.vstack(
                [
                    all_possible_locations,
                    np.array(
                        self.graph_memory.get_locations_in_graph(
                            graph_id, input_channel
                        )
                    ),
                ]
            )
            all_possible_rotations = np.vstack([all_possible_rotations, r])

        return all_possible_locations[1:], all_possible_rotations[1:]

    def _threshold_possible_matches(self, x_percent_scale_factor=1.0):
        """Return possible matches based on evidence threshold.

        Args:
            x_percent_scale_factor: If desired, can check possible matches using a
                scaled threshold; can be used to e.g. check whether hypothesis-testing
                policy should focus on descriminating a single object's pose, vs.
                between different object IDs, when we are half-way to the threshold
                required for classification; "mod" --> modifier
                By default set to identity and has no effect
                Should be bounded 0:1.0

        Returns:
            The possible matches.
        """
        if len(self.graph_memory) == 0:
            logging.info("no objects in memory yet.")
            return []
        graph_ids, graph_evidences = self.get_evidence_for_each_graph()
        # median_ge = np.median(graph_evidences)
        mean_ge = np.mean(graph_evidences)
        max_ge = np.max(graph_evidences)
        std_ge = np.std(graph_evidences)

        if (std_ge > 0.1) or (max_ge < 0):
            # If all evidences are below 0, return no possible objects
            # th = max_ge - std_ge if max_ge > 0 else 0
            x_percent_of_max = (
                max_ge / 100 * self.x_percent_threshold * x_percent_scale_factor
            )
            th = max_ge - x_percent_of_max if max_ge > 0 else 0
            pm = [graph_ids[i] for i, ge in enumerate(graph_evidences) if ge > th]
            logging.debug(f"evidences for each object: {graph_evidences}")
            logging.debug(
                f"mean evidence: {np.round(mean_ge, 3)}, std: {np.round(std_ge, 3)}"
                f" -> th={th}"
            )
        elif len(self.graph_memory) == 1:
            # Making it a bit more explicit what happens if we only have one graph
            # in memory. In this case we basically recognize the object if the evidence
            # is > object_evidence_threshold and we have resolved a pose.
            # NOTE: This may not be the best way to handle this and can cause problems
            # when learning from scratch.
            # TODO: Figure out a better way to deal with incomplete and few objects in
            # memory
            pm = [graph_ids[i] for i, ge in enumerate(graph_evidences) if ge >= 0]
        else:  # objects are about equally likely
            pm = graph_ids
        return pm

    def _get_mlh_dict_from_id(self, graph_id, mlh_id):
        """Return dict with mlh information for given graph_id and mlh_id.

        Args:
            graph_id: id of graph
            mlh_id: Int index of most likely hypothesis within graph

        Returns:
            The most likely hypothesis dictionary.
        """
        mlh_dict = {
            "graph_id": graph_id,
            "location": self.possible_locations[graph_id][mlh_id],
            "rotation": Rotation.from_matrix(self.possible_poses[graph_id][mlh_id]),
            "scale": self.get_object_scale(graph_id),
            "evidence": self.evidence[graph_id][mlh_id],
        }
        return mlh_dict

    def _calculate_most_likely_hypothesis(self, graph_id=None):
        """Return pose with highest evidence count.

        Args:
            graph_id: If provided, find mlh pose for this object. If graph_id is None
                look through all objects and finds most likely one.

        Returns dict with keys: object_id, location, rotation, scale, evidence
        """
        mlh = {}
        if graph_id is not None:
            mlh_id = np.argmax(self.evidence[graph_id])
            mlh = self._get_mlh_dict_from_id(graph_id, mlh_id)
        else:
            highest_evidence_so_far = -np.inf
            for graph_id in self.get_all_known_object_ids():
                mlh_id = np.argmax(self.evidence[graph_id])
                evidence = self.evidence[graph_id][mlh_id]
                if evidence > highest_evidence_so_far:
                    mlh = self._get_mlh_dict_from_id(graph_id, mlh_id)
                    highest_evidence_so_far = evidence
            if not mlh:  # No objects in memory
                mlh = self.current_mlh
                mlh["graph_id"] = "new_object0"
            logging.info(
                f"current most likely hypothesis: {mlh['graph_id']} "
                f"with evidence {np.round(mlh['evidence'],2)}"
            )
        return mlh

    def _get_evidence_update_threshold(self, graph_id):
        """Determine how much evidence a hypothesis should have to be updated.

        Returns:
            The evidence update threshold.

        Raises:
            Exception: If `self.evidence_update_threshold` is not in the allowed
                values
        """
        if type(self.evidence_update_threshold) in [int, float]:
            return self.evidence_update_threshold
        elif self.evidence_update_threshold == "mean":
            return np.mean(self.evidence[graph_id])
        elif self.evidence_update_threshold == "median":
            return np.median(self.evidence[graph_id])
        elif isinstance(
            self.evidence_update_threshold, str
        ) and self.evidence_update_threshold.endswith("%"):
            percentage_str = self.evidence_update_threshold.strip("%")
            percentage = float(percentage_str)
            assert (
                percentage >= 0 and percentage <= 100
            ), "Percentage must be between 0 and 100"
            max_global_evidence = self.current_mlh["evidence"]
            x_percent_of_max = max_global_evidence * (percentage / 100)
            return max_global_evidence - x_percent_of_max
        elif self.evidence_update_threshold == "x_percent_threshold":
            max_global_evidence = self.current_mlh["evidence"]
            x_percent_of_max = max_global_evidence / 100 * self.x_percent_threshold
            return max_global_evidence - x_percent_of_max
        elif self.evidence_update_threshold == "all":
            return np.min(self.evidence[graph_id])
        else:
            raise Exception(
                "evidence_update_threshold not in "
                "[int, float, '[int]%', 'mean', 'median', 'all', 'x_percent_threshold']"
            )

    def _get_node_distance_weights(self, distances):
        node_distance_weights = (
            self.max_match_distance - distances
        ) / self.max_match_distance
        return node_distance_weights

    # ----------------------- Logging --------------------------
    def _add_votes_to_buffer_stats(self, vote_data):
        # Do we want to store this? will probably just clutter.
        # self.buffer.update_stats(vote_data, update_time=False)
        pass

    def _add_detailed_stats(self, stats, get_rotations):
        stats["possible_locations"] = self.possible_locations
        if get_rotations:
            stats["possible_rotations"] = self.get_possible_poses(as_euler=False)
        stats["evidences"] = self.evidence
        stats["symmetry_evidence"] = self.symmetry_evidence
        return stats


class EvidenceGraphMemory(GraphMemory):
    """Custom GraphMemory that stores GridObjectModel instead of GraphObjectModel."""

    def __init__(
        self,
        max_nodes_per_graph,
        max_graph_size,
        num_model_voxels_per_dim,
        *args,
        **kwargs,
    ):
        super(EvidenceGraphMemory, self).__init__(*args, **kwargs)

        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_graph_size = max_graph_size
        self.num_model_voxels_per_dim = num_model_voxels_per_dim

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        possible_matches = self.get_memory_ids()
        possible_locations = {}
        for graph_id in possible_matches:
            # Initialize vertical shape of np array so we can easily stack it. Will be
            # removed after concatenating loop.
            all_locations_for_graph = np.zeros(3)
            for input_channel in self.get_input_channels_in_graph(graph_id):
                # All locations stored in graph
                all_locations_for_graph = np.vstack(
                    [
                        all_locations_for_graph,
                        self.get_locations_in_graph(graph_id, input_channel),
                    ]
                )
            possible_locations[graph_id] = all_locations_for_graph[1:]
        return possible_matches, possible_locations

    def get_rotation_features_at_all_nodes(self, graph_id, input_channel):
        """Get rotation features from all N nodes. shape=(N, 3, 3).

        Returns:
            The rotation features from all N nodes. shape=(N, 3, 3).
        """
        all_node_r_features = self.get_features_at_node(
            graph_id,
            input_channel,
            self.get_graph_node_ids(graph_id, input_channel),
            feature_keys=["pose_vectors"],
        )
        node_directions = all_node_r_features["pose_vectors"]
        num_nodes = len(node_directions)
        node_directions = node_directions.reshape((num_nodes, 3, 3))
        return node_directions

    # ------------------ Logging & Saving ----------------------

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add pretrained graph to memory.

        Initializes GridObjectModel and calls set_graph.

        Args:
            model: new model to be added to memory
            graph_id: id of graph that should be added

        """
        self.models_in_memory[graph_id] = dict()
        for input_channel in model.keys():
            channel_model = model[input_channel]
            try:
                if type(channel_model) == GraphObjectModel:
                    # When loading a model trained with a different LM, need to convert
                    # it to the GridObjectModel (with use_orginal_graph == True)
                    loaded_graph = channel_model._graph
                    channel_model = self._initialize_model_with_graph(
                        graph_id, loaded_graph
                    )

                logging.info(f"Loaded {model} for {input_channel}")
                self.models_in_memory[graph_id][input_channel] = channel_model
            except GridTooSmallError:
                logging.info(
                    "Grid too small for given locations. Not adding to memory."
                )

    def _initialize_model_with_graph(self, graph_id, graph):
        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        # Keep benchmark results constant by still using orginal graph for
        # matching when loading pretrained models.
        model.use_orginal_graph = True
        model.set_graph(graph)
        return model

    def _build_graph(self, locations, features, graph_id, input_channel):
        """Build a graph from a list of features at locations and add to memory.

        This initialzes a new GridObjectModel and calls model.build_graph.

        Args:
            locations: List of x,y,z locations.
            features: List of features.
            graph_id: name of new graph.
            input_channel: ?
        """
        logging.info(f"Adding a new graph to memory.")

        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        try:
            model.build_model(locations=locations, features=features)

            if graph_id not in self.models_in_memory:
                self.models_in_memory[graph_id] = dict()
            self.models_in_memory[graph_id][input_channel] = model

            logging.info(f"Added new graph with id {graph_id} to memory.")
            logging.info(model)
        except GridTooSmallError:
            logging.info(
                "Grid too small for given locations. Not building a model "
                f"for {graph_id}"
            )

    def _extend_graph(
        self,
        locations,
        features,
        graph_id,
        input_channel,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
        object_scale,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x,y,z locations.
            features: Features observed at the locations.
            graph_id: ID of the existing graph.
            input_channel: ?
            object_location_rel_body: location of the sensor in body reference frame
            location_rel_model: location of sensor in model reference frame
            object_rotation: rotation of the sensed object relative to the model
            object_scale: scale of the object relative to the model of it
        """
        logging.info(f"Updating existing graph for {graph_id}")

        try:
            self.models_in_memory[graph_id][input_channel].update_model(
                locations=locations,
                features=features,
                location_rel_model=location_rel_model,
                object_location_rel_body=object_location_rel_body,
                object_rotation=object_rotation,
            )
            logging.info(
                f"Extended graph {graph_id} with new points. New model:\n"
                f"{self.models_in_memory[graph_id]}"
            )
        except GridTooSmallError:
            logging.info("Grid too small for given locations. Not updating model.")

    # ------------------------ Helper --------------------------

    # ----------------------- Logging --------------------------
