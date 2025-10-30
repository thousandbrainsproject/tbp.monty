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

import logging
import threading
import time

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    ChannelHypotheses,
    Hypotheses,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    DefaultHypothesesUpdater,
    HypothesesUpdater,
    HypothesesUpdaterTelemetry,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    ConsistentHypothesesIds,
    evidence_update_threshold,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    add_pose_features_to_tolerances,
    get_scaled_evidences,
)

logger = logging.getLogger(__name__)


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
        evidence_threshold_config: How to decide which hypotheses
            should be updated. When this parameter is either '[int]%' or
            'x_percent_threshold', then this parameter is applied to the evidence
            for the Most Likely Hypothesis (MLH) to determine a minimum evidence
            threshold in order for other hypotheses to be updated. Any hypotheses
            falling below the resulting evidence threshold do not get updated. The
            other options set a fixed threshold that does not take MLH evidence into
            account. In [int, float, '[int]%', 'mean', 'median', 'all',
            'x_percent_threshold']. Defaults to 'all'.
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
            # TODO: should we use a separate threshold for within and between objects?
            If this value is larger, the model is usually more robust to noise and
            reaches a better performance but also requires a lot more steps to reach a
            terminal condition, especially if there are many similar object in the data
            set.
        path_similarity_threshold: How similar do paths have to be to be
            considered the same in the terminal condition check.
        pose_similarity_threshold: difference between two poses to be considered
            unique when checking for the terminal condition (in radians).
        required_symmetry_evidence: number of steps with unchanged possible poses
            to classify an object as symmetric and go into terminal condition.

    Model Attributes:
        graph_delta_thresholds: Thresholds used to compare nodes in the graphs being
            learned, and thereby whether to include a new point or not. By default,
            we only consider the distance between points, using a threshold
            of 0.001 (determined in remove_close_points). Can also specify
            thresholds based on e.g. surface normal angle difference, or principal
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
        hypotheses_updater_class: The type of hypotheses updater to associate with the
            LM.
        hypotheses_updater_args: Dictionary of configuration parameters for the
            hypotheses updater.

    Debugging Attributes:
        use_multithreading: Whether to calculate evidence updates for different
            objects in parallel using multithreading. This can be done since the
            updates to different objects are completely independent of each other. In
            general it is recommended to use this but it can be useful to turn it off
            for debugging purposes.
    """

    def __init__(
        self,
        max_match_distance,
        tolerances: dict,
        feature_weights: dict,
        feature_evidence_increment=1,
        evidence_threshold_config: float | str = "all",
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
        hypotheses_updater_class: type[HypothesesUpdater] = DefaultHypothesesUpdater,
        hypotheses_updater_args: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        kwargs["initialize_base_modules"] = False
        super().__init__(*args, **kwargs)
        # --- LM components ---
        self.graph_memory = EvidenceGraphMemory(
            graph_delta_thresholds=graph_delta_thresholds,
            max_nodes_per_graph=max_nodes_per_graph,
            max_graph_size=max_graph_size,
            num_model_voxels_per_dim=num_model_voxels_per_dim,
        )
        if gsg_class is not None:
            gsg_args = gsg_args or {}
            self.gsg = gsg_class(self, **gsg_args)
            self.gsg.reset()
        else:
            self.gsg = None
        # --- Matching Params ---
        self.max_match_distance = max_match_distance
        self.tolerances = tolerances
        self.feature_evidence_increment = feature_evidence_increment
        self.evidence_threshold_config = evidence_threshold_config
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

        # Dictionary with graph_ids as keys. For each graph we initialize a set of
        # hypotheses at the first step of an episode. Each hypothesis has an evidence
        # count associated with it which is stored here.
        # self.possible_locations and self.possible_poses have the same structure and
        # length as self.evidence and store the corresponding hypotheses.
        self.evidence: dict[str, np.ndarray] = {}
        self.possible_locations: dict[str, np.ndarray] = {}
        self.possible_poses: dict[str, np.ndarray] = {}

        # A dictionary from graph_id to instances of `ChannelMapper`.
        self.channel_hypothesis_mapping: dict[str, ChannelMapper] = {}

        self.current_mlh = {
            "graph_id": "no_observations_yet",
            "location": [0, 0, 0],
            "rotation": Rotation.from_euler("xyz", [0, 0, 0]),
            "scale": 1,
            "evidence": 0,
        }

        if hypotheses_updater_args is None:
            hypotheses_updater_args = {}
        # Every HypothesesUpdater gets at least the following arguments because they are
        # either constructed or edited in the constructor, or they are shared with the
        # learning module.
        hypotheses_updater_args.update(
            evidence_threshold_config=self.evidence_threshold_config,
            feature_evidence_increment=self.feature_evidence_increment,
            feature_weights=self.feature_weights,
            graph_memory=self.graph_memory,
            max_match_distance=self.max_match_distance,
            past_weight=self.past_weight,
            present_weight=self.present_weight,
            tolerances=self.tolerances,
        )
        self.hypotheses_updater = hypotheses_updater_class(**hypotheses_updater_args)
        self.hypotheses_updater_telemetry: HypothesesUpdaterTelemetry = {}

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def reset(self):
        """Reset evidence count and other variables."""
        # Now here, as opposed to the displacement and feature-location LMs,
        # possible_matches is a list of IDs, not a dictionary with the object graphs.
        self.possible_matches = self.graph_memory.get_initial_hypotheses()

        if self.tolerances is not None:
            # TODO H: Differentiate between features from different input channels
            # TODO: could do this in the object model class
            self.graph_memory.initialize_feature_arrays()
        self.symmetry_evidence = 0
        self.last_possible_hypotheses = None
        self.channel_hypothesis_mapping = {}

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
            logger.debug("Updating possible matches after vote")
            self.possible_matches = self._threshold_possible_matches()
            self.current_mlh = self._calculate_most_likely_hypothesis()

            self._add_votes_to_buffer_stats(vote_data)

    def send_out_vote(self) -> dict | None:
        """Send out hypotheses and the evidence for them.

        Votes are a dict and contain the following:
            pose_hypotheses: locations (V, 3) and rotations (V, 3, 3)
            pose_evidence: Evidence (V) for each location-rotation pair in the
                    pose hypotheses. Scaled into range [-1, 1] where 1 is the
                    hypothesis with the largest evidence in this LM and -1 the
                    one with the smallest evidence. When thresholded, pose_evidence
                    will be in range [self.vote_evidence_threshold, 1]
            sensed_pose_rel_body: sensed location and rotation of the input to this
                    LM. Rotation is represented by the pose vectors (surface normal and
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
            Dictionary with possible states and sensed poses relative to the body, or
            None if we don't want the LM to vote.
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

            possible_states = {}
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
        logger.debug(f"terminal state: {terminal_state}")
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
            self.last_possible_hypotheses = (
                self.hypotheses_updater.remap_hypotheses_ids_to_present(
                    self.last_possible_hypotheses
                )
            )
            symmetry_detected = self._check_for_symmetry(
                object_id,
                possible_object_hypotheses_ids,
                # Don't increment symmetry counter if LM didn't process observation
                increment_evidence=self.buffer.get_last_obs_processed(),
            )
            self.last_possible_hypotheses = ConsistentHypothesesIds(
                hypotheses_ids=possible_object_hypotheses_ids,
                channel_sizes=self.channel_hypothesis_mapping[object_id].channel_sizes,
                graph_id=object_id,
            )

            if pose_is_unique or symmetry_detected:
                r_inv = mlh["rotation"].inv()
                r_euler = mlh["rotation"].inv().as_euler("xyz", degrees=True)
                r_euler = np.round(r_euler, 3) % 360

                pose_and_scale = np.concatenate(
                    [mlh["location"], r_euler, [mlh["scale"]]]
                )
                logger.debug(f"(location, rotation, scale): {pose_and_scale}")
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
                            self.last_possible_hypotheses.hypotheses_ids
                        ],
                        "symmetric_locations": self.possible_locations[object_id][
                            self.last_possible_hypotheses.hypotheses_ids
                        ],
                    }
                    self.buffer.add_overall_stats(symmetry_stats)
                return pose_and_scale
            else:
                logger.debug(f"object {object_id} detected but pose not resolved yet.")
                return None
        else:
            self.last_possible_hypotheses = None
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

        # If all hypothesis spaces are empty return None for both mlh ids. The gsg will
        # not generate a goal state.
        if len(graph_ids) == 0:
            return None, None

        # If we have a single hypothesis space, return the second object id as None.
        # The gsg will focus on pose to generate a goal state.
        if len(graph_ids) == 1:
            return graph_ids[0], None

        # Note the indices below will be ordered with the 2nd MLH appearing first, and
        # the 1st MLH appearing second.
        top_indices = np.argsort(graph_evidences)[-2:]

        top_id = graph_ids[top_indices[1]]
        second_id = graph_ids[top_indices[0]]

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
            all_poses = {}
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
            logger.debug(
                f"possible hpids: {possible_object_hypotheses_ids} for {object_id}"
            )
            logger.debug(f"hpid evidence is > {max_obj_evidence} - {x_percent_of_max}")
            return possible_object_hypotheses_ids

    def get_evidence_for_each_graph(self):
        """Return maximum evidence count for a pose on each graph."""
        graph_ids = self.get_all_known_object_ids()
        if graph_ids[0] not in self.evidence.keys():
            return ["patch_off_object"], [0]

        available_graph_ids = []
        available_graph_evidences = []
        for graph_id in graph_ids:
            if len(self.hyp_evidences_for_object(graph_id)):
                available_graph_ids.append(graph_id)
                available_graph_evidences.append(np.max(self.evidence[graph_id]))

        return available_graph_ids, np.array(available_graph_evidences)

    def get_all_evidences(self):
        """Return evidence for each pose on each graph (pointer)."""
        return self.evidence

    def hyp_evidences_for_object(self, object_id):
        """Return evidences for a specific object_id."""
        return self.evidence[object_id]

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
            stats = self._add_detailed_stats(stats)
        return stats

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

    def _update_evidence(
        self,
        features: dict,
        displacements: dict | None,
        graph_id: str,
    ) -> None:
        """Update evidence based on sensor displacement and sensed features.

        Updates existing hypothesis space or initializes a new hypothesis space
        if one does not exist (i.e., at the beginning of the episode). Updating the
        hypothesis space includes displacing the hypotheses possible locations, as well
        as updating their evidence scores. This process is repeated for each input
        channel in the graph.

        Args:
            features: input features
            displacements: given displacements
            graph_id: identifier of the graph being updated
        """
        start_time = time.time()

        # Initialize a `ChannelMapper` to keep track of input channel range
        # of hypotheses for a specific graph_id
        if graph_id not in self.channel_hypothesis_mapping:
            self.channel_hypothesis_mapping[graph_id] = ChannelMapper()
            self.evidence[graph_id] = np.array([])
            self.possible_locations[graph_id] = np.array([])
            self.possible_poses[graph_id] = np.array([])

        # Calculate the evidence_update_threshold
        update_threshold = evidence_update_threshold(
            self.evidence_threshold_config,
            self.x_percent_threshold,
            max_global_evidence=self.current_mlh["evidence"],
            evidence_all_channels=self.evidence[graph_id],
        )

        hypotheses_updates, hypotheses_update_telemetry = (
            self.hypotheses_updater.update_hypotheses(
                hypotheses=Hypotheses(
                    evidence=self.evidence[graph_id],
                    locations=self.possible_locations[graph_id],
                    poses=self.possible_poses[graph_id],
                ),
                features=features,
                displacements=displacements,
                graph_id=graph_id,
                mapper=self.channel_hypothesis_mapping[graph_id],
                evidence_update_threshold=update_threshold,
            )
        )

        if hypotheses_update_telemetry is not None:
            self.hypotheses_updater_telemetry[graph_id] = hypotheses_update_telemetry

        if not hypotheses_updates:
            return

        for update in hypotheses_updates:
            self._set_hypotheses_in_hpspace(graph_id=graph_id, new_hypotheses=update)

        end_time = time.time()

        logger_msg = (
            f"evidence update for {graph_id} took "
            f"{np.round(end_time - start_time, 2)} seconds."
        )
        graph_evidence = self.hyp_evidences_for_object(graph_id)
        if len(graph_evidence):
            assert not np.isnan(np.max(self.evidence[graph_id])), (
                "evidence contains NaN."
            )
            logger_msg += (
                f" New max evidence: {np.round(np.max(self.evidence[graph_id]), 3)}"
            )
        logger.debug(logger_msg)

    def _set_hypotheses_in_hpspace(
        self,
        graph_id: str,
        new_hypotheses: ChannelHypotheses,
    ) -> None:
        """Updates the hypothesis space for a given input channel in a graph.

        This function updates the hypothesis space (for a specific graph and input
        channel) with a new set of locations, rotations and evidence scores.
            - If the hypothesis space does not exist for any input channel, a new one
                is initialized
            - If the hypothesis space only exists for other channels, a new channel is
                created with the mean evidence scores of the existing channels
            - If the hypothesis space exists for the given input channel, the new space
                replaces the existing hypothesis space

        Args:
            graph_id: The ID of the current graph to update.
            new_hypotheses: The new hypotheses to set. These are the
                sets of location, pose, and evidence after applying movements to the
                possible locations and updating their evidence scores. These could also
                refer to newly initialized hypotheses if a hypothesis space did not
                exist.
        """
        # Extract channel mapper
        mapper = self.channel_hypothesis_mapping[graph_id]
        new_evidence = new_hypotheses.evidence

        if new_hypotheses.input_channel not in mapper.channels:
            # If there are currently no channels in the mapper, initialize the
            # space with empty arrays of the correct shapes.
            if len(mapper.channels) == 0:
                self.possible_locations[graph_id] = np.empty((0, 3))
                self.possible_poses[graph_id] = np.empty((0, 3, 3))
                self.evidence[graph_id] = np.empty((0,))

            # If there exists other channels, add current mean evidence to give the
            # new hypotheses a fighting chance.
            # TODO H: Test mean vs. median here.
            else:
                current_mean_evidence = np.mean(self.evidence[graph_id])
                new_evidence = new_evidence + current_mean_evidence

            # Add a mapper channel to be updated with the new data. The mapper will
            # be later resized to `len(new_evidence)` after we update the hypothesis
            # space.
            mapper.add_channel(new_hypotheses.input_channel, 0)

        # The mapper update function calls below automatically resize the
        # arrays they update. Afterward, we must update the channel indices
        # in the mapper via resize_channel_to to stay in sync with
        # the now resized arrays. We do not resize before array updates
        # because then, during the update, the indices would not correspond
        # to the data in the arrays.
        self.possible_locations[graph_id] = mapper.update(
            self.possible_locations[graph_id],
            new_hypotheses.input_channel,
            new_hypotheses.locations,
        )
        self.possible_poses[graph_id] = mapper.update(
            self.possible_poses[graph_id],
            new_hypotheses.input_channel,
            new_hypotheses.poses,
        )
        self.evidence[graph_id] = mapper.update(
            self.evidence[graph_id],
            new_hypotheses.input_channel,
            new_evidence,
        )

        mapper.resize_channel_to(new_hypotheses.input_channel, len(new_evidence))

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
        logger.debug(f"{possible_locations.shape[0]} possible locations")

        center_location = np.mean(possible_locations, axis=0)
        distances_to_center = np.linalg.norm(
            possible_locations - center_location, axis=1
        )
        location_unique = np.max(distances_to_center) < self.path_similarity_threshold
        if location_unique:
            logger.info(
                "all possible locations are in radius "
                f"{self.path_similarity_threshold} of {center_location}"
            )

        possible_rotations = np.array(self.possible_poses[graph_id])[
            possible_object_hypotheses_ids
        ]
        logger.debug(f"{possible_rotations.shape[0]} possible rotations")

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

    def _check_for_symmetry(
        self, object_id, possible_object_hypotheses_ids, increment_evidence
    ):
        """Check whether the most likely hypotheses stayed the same over the past steps.

        Since the definition of possible_object_hypotheses is a bit murky and depends
        on how we set an evidence threshold we check the set overlap here and see if at
        least 90% of the current hypotheses were also possible on the last step. I'm
        not sure if this is the best way to check for symmetry...

        Args:
            object_id: identifier of the object being checked for symmetry
            possible_object_hypotheses_ids: List of IDs of all possible hypotheses.
            increment_evidence: Whether to increment symmetry evidence or not. We
                may want this to be False for example if we did not receive a new
                observation.

        Returns:
            Whether symmetry was detected.
        """
        if self.last_possible_hypotheses is None:
            return False  # need more steps to meet symmetry condition
        logger.debug(
            f"\n\nchecking for symmetry for hp ids {possible_object_hypotheses_ids}"
            f" with last ids {self.last_possible_hypotheses}"
        )
        if increment_evidence and self.last_possible_hypotheses.graph_id == object_id:
            previous_hyps = set(self.last_possible_hypotheses.hypotheses_ids)
            current_hyps = set(possible_object_hypotheses_ids)
            hypothesis_overlap = previous_hyps.intersection(current_hyps)
            if len(hypothesis_overlap) / len(current_hyps) > 0.9:
                # at least 90% of current possible ids were also in previous ids
                logger.info("added symmetry evidence")
                self.symmetry_evidence += 1
            else:  # has to be consecutive
                self.symmetry_evidence = 0

        if self._enough_symmetry_evidence_accumulated():
            logger.info(
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
        id_feature = sum(ord(i) for i in object_id)
        return id_feature

    def _fill_feature_weights_with_default(self, default: int) -> None:
        for input_channel, channel_tolerances in self.tolerances.items():
            if input_channel not in self.feature_weights.keys():
                self.feature_weights[input_channel] = {}
            for key, tolerance in channel_tolerances.items():
                if key not in self.feature_weights[input_channel].keys():
                    if hasattr(tolerance, "shape"):
                        shape = tolerance.shape
                    elif hasattr(tolerance, "__len__"):
                        shape = len(tolerance)
                    else:
                        shape = 1
                    default_weights = np.ones(shape) * default
                    logger.debug(
                        f"adding {key} to feature_weights with value {default_weights}"
                    )
                    self.feature_weights[input_channel][key] = default_weights

    def _threshold_possible_matches(self, x_percent_scale_factor=1.0):
        """Return possible matches based on evidence threshold.

        Args:
            x_percent_scale_factor: If desired, can check possible matches using a
                scaled threshold; can be used to e.g. check whether hypothesis-testing
                policy should focus on discriminating a single object's pose, vs.
                between different object IDs, when we are half-way to the threshold
                required for classification; "mod" --> modifier
                By default set to identity and has no effect
                Should be bounded 0:1.0

        Returns:
            The possible matches.
        """
        if len(self.graph_memory) == 0:
            logger.info("no objects in memory yet.")
            return []

        graph_ids, graph_evidences = self.get_evidence_for_each_graph()

        if len(graph_ids) == 0:
            logger.info("All hypothesis spaces are empty. No possible matches.")
            return []

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
            logger.debug(f"evidences for each object: {graph_evidences}")
            logger.debug(
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
            graph_evidence = self.hyp_evidences_for_object(graph_id)
            if len(graph_evidence):
                mlh_id = np.argmax(graph_evidence)
                mlh = self._get_mlh_dict_from_id(graph_id, mlh_id)
        else:
            highest_evidence_so_far = -np.inf
            for graph_id in self.get_all_known_object_ids():
                graph_evidence = self.hyp_evidences_for_object(graph_id)
                if len(graph_evidence):
                    mlh_id = np.argmax(graph_evidence)
                    evidence = graph_evidence[mlh_id]
                    if evidence > highest_evidence_so_far:
                        mlh = self._get_mlh_dict_from_id(graph_id, mlh_id)
                        highest_evidence_so_far = evidence

        if not mlh:  # No objects in memory
            mlh = self.current_mlh
            mlh["graph_id"] = "new_object0"
        logger.info(
            f"current most likely hypothesis: {mlh['graph_id']} "
            f"with evidence {np.round(mlh['evidence'], 2)}"
        )
        return mlh

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

    def _add_detailed_stats(self, stats):
        # Save possible poses once since they don't change during episode
        get_rotations = False
        if "possible_rotations" not in self.buffer.stats.keys():
            get_rotations = True

        stats["possible_locations"] = self.possible_locations
        if get_rotations:
            stats["possible_rotations"] = self.get_possible_poses(as_euler=False)
        stats["evidences"] = self.evidence
        stats["symmetry_evidence"] = self.symmetry_evidence
        return stats
