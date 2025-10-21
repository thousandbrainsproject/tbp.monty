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

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from tbp.monty.frameworks.models.graph_matching import GraphLM, GraphMemory
from tbp.monty.frameworks.utils.graph_matching_utils import (
    add_pose_features_to_tolerances,
    get_initial_possible_poses,
    get_unique_paths,
    possible_sensed_directions,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
    get_angle,
    get_unique_rotations,
    rotate_pose_dependent_features,
)

logger = logging.getLogger(__name__)


class FeatureGraphLM(GraphLM):
    """Learning module that uses features at locations to recognize objects."""

    # FIXME: hardcoding the number of LMs that we expect to be voting with
    NUM_OTHER_LMS = 4

    def __init__(
        self,
        max_match_distance,
        tolerances,
        path_similarity_threshold=0.1,
        pose_similarity_threshold=0.35,
        required_symmetry_evidence=5,
        graph_delta_thresholds=None,
        initial_possible_poses="informed",
        umbilical_num_poses=8,
    ):
        """Initialize Learning Module.

        Args:
            max_match_distance: Maximum distance of a tested and stored location to
                be matched.
            tolerances: How much can each observed feature deviate from the stored
                features to still be considered a match.
            graph_delta_thresholds: Thresholds used to compare nodes in the graphs
                being learned, and thereby whether to include a new point or not. By
                default, we only consider the distance between points, using a
                threshold of 0.001 (determined in remove_close_points). Can also
                specify thresholds based on e.g. surface normal angle difference, or
                principal curvature magnitude difference.
            path_similarity_threshold: How similar do paths have to be
                considered the same in the terminal condition check.
            pose_similarity_threshold: difference between two poses to be considered
                unique when checking for the terminal condition (in radians).
            required_symmetry_evidence: number of steps with unchanged possible poses
                to classify an object as symmetric and go into terminal condition.
            initial_possible_poses: initial possible poses that should be tested for.
                In ["uniform", "informed", list]. default = "informed".
            umbilical_num_poses: Number of samples rotations in the direction
                of the plane perpendicular to the surface normal.
        """
        super().__init__()
        self.graph_memory = FeatureGraphMemory(
            graph_delta_thresholds=graph_delta_thresholds,
        )
        # make sure we extract pose dependent features because they
        # are nescessary for the algorithm to work.
        self.tolerances = add_pose_features_to_tolerances(tolerances)
        self.max_match_distance = max_match_distance
        self.path_similarity_threshold = path_similarity_threshold
        self.pose_similarity_threshold = pose_similarity_threshold
        self.required_symmetry_evidence = required_symmetry_evidence

        # TODO: not ideal solution
        self.graph_memory.features_to_use = self.tolerances

        self.initial_possible_poses = get_initial_possible_poses(initial_possible_poses)
        self.umbilical_num_poses = umbilical_num_poses
        self.possible_poses = {}
        self.last_unique_poses = None
        self.last_num_unique_locations = None

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def reset(self):
        """Call this before each episode."""
        (
            self.possible_matches,
            self.possible_paths,
            self.possible_poses,
        ) = self.graph_memory.get_initial_hypotheses()

        if self.tolerances is not None:
            self.graph_memory.initialize_feature_arrays()
        self.symmetry_evidence = 0
        self.last_unique_poses = None
        self.last_num_unique_locations = None

    def send_out_vote(self):
        """Send out list of objects that are not possible matches.

        By sending out the negative matches we avoid the problem that
        every LM needs to know about the same objects. We could think of
        this as more of an inhibitory signal (I know it can't be this
        object so you all don't need to check that anymore).

        Returns:
            List of objects that are not possible matches.
        """
        possible_matches = self.get_possible_matches()
        all_objects = self.get_all_known_object_ids()
        object_id_vote = {}
        for obj in all_objects:
            object_id_vote[obj] = obj in possible_matches
        logger.info(
            f"PM: {possible_matches} out of all: {all_objects} "
            f"-> vote: {object_id_vote}"
        )
        possible_locations = self.get_possible_locations()
        possible_poses = self.get_possible_poses(as_euler=False)
        sensed_pose = self.buffer.get_current_pose(input_channel="first")
        vote = {
            "object_id_vote": object_id_vote,
            "location_vote": possible_locations,
            "rotation_vote": possible_poses,
            "sensed_pose_rel_body": sensed_pose,
        }
        return vote

    def receive_votes(self, vote_data):
        """Use votes to remove objects and poses from possible matches.

        NOTE: Add object back into possible matches if majority of other modules
                think it is correct? Could help with dealing with noise but may
                also prevent LMs from narrowing down quickly. Since we are not
                working with this LM anymore, we probably wont add that.

        Args:
            vote_data: positive and negative votes on object IDs + positive
                votes for locations and rotations on the object.
        """
        if (vote_data is not None) and (
            self.buffer.get_num_observations_on_object() > 0
        ):
            current_possible_matches = self.get_possible_matches()
            for possible_obj in current_possible_matches:
                if (
                    vote_data["neg_object_id_votes"][possible_obj]
                    > vote_data["pos_object_id_votes"][possible_obj]
                ):
                    logger.info(f"Removing {possible_obj} from matches after vote.")
                    self._remove_object_from_matches(possible_obj)

                # Check that object is still in matches after ID update
                if possible_obj in self.possible_matches:
                    if vote_data["pos_location_votes"][possible_obj].shape[0] < (
                        self.NUM_OTHER_LMS
                    ):
                        k = vote_data["pos_location_votes"][possible_obj].shape[0]
                        logger.info(f"only received {k} votes")
                    else:
                        # k should not be > num_lms - 1
                        k = self.NUM_OTHER_LMS
                    vote_location_tree = KDTree(
                        vote_data["pos_location_votes"][possible_obj],
                        leaf_size=2,
                    )
                    removed_locations = np.zeros((1, 3))
                    # print("updating possible locations on model")
                    for path_id, path in reversed(
                        list(enumerate(self.possible_paths[possible_obj]))
                    ):
                        location = path[-1]
                        dists, _ = vote_location_tree.query([location], k=k)
                        # print(f"distances of nearest votes: {dists}")
                        # TODO: check pose vote as well.
                        # TODO: adapt this to number of LMs/received votes
                        # vote distance needs to be larger to deal with case where
                        # agent step size used to collect observations for model during
                        # learning is larger than max_match_distance -> model is
                        # sampled less densely than we vote.
                        # TODO: determine this more flexibly.
                        if dists[0][k - 1] > self.max_match_distance * 10:
                            self.possible_paths[possible_obj].pop(path_id)
                            self.possible_poses[possible_obj].pop(path_id)
                            removed_locations = np.vstack([removed_locations, location])
                    logger.info(
                        f"removed {removed_locations.shape[0] - 1} locations from "
                        f"possible matches for {possible_obj}"
                    )
                    # NOTE: could also use votes to add hypotheses -> increase
                    # robustness, especially with possible poses.

                    # Remove object if after location vote no locations are left.
                    if len(self.possible_paths[possible_obj]) == 0:
                        self._remove_object_from_matches(possible_obj)

            self._add_votes_to_buffer_stats(vote_data)

    # ------------------ Getters & Setters ---------------------
    def get_unique_pose_if_available(self, object_id):
        """Get the pose of an object if we know it.

        Scale is not implemented.

        Returns:
            The pose of the object if we know it.
        """
        pose_and_scale = None

        r_euler, r = self.get_object_rotation(
            object_id,
            get_reverse_r=True,  # since we rotate the displacement and not model
        )
        # is None of pose is not resolved yet. A pose is resolved if we either have
        # one possible location and rotation (with tolerance) or detected symmetry.
        if r_euler is not None:
            possible_paths = self.get_possible_paths()[object_id]
            detected_path = possible_paths[0]
            model_locs = detected_path
            current_model_loc = model_locs[-1]
            scale = self.get_object_scale(object_id)  # NOTE: Scale doesn't work for FM
            pose_and_scale = np.concatenate([current_model_loc, r_euler[0], [scale]])
            logger.debug(f"(location, rotation, scale): {pose_and_scale}")

            # Update own state
            self.detected_pose = pose_and_scale
            self.detected_rotation_r = r[0]
            # Update buffer stats
            lm_episode_stats = {
                "detected_path": detected_path,
                "detected_location_on_model": current_model_loc,
                "detected_location_rel_body": self.buffer.get_current_location(
                    input_channel="first"
                ),
                "detected_rotation": r_euler[0],
                "detected_rotation_quat": [rot.as_quat() for rot in r],
                "detected_scale": scale,
            }
            self.buffer.add_overall_stats(lm_episode_stats)
            if self._enough_symmetry_evidence_accumulated():
                symmetry_stats = {
                    "symmetric_rotations": self.last_unique_poses,
                    "symmetric_locations": np.array(possible_paths)[:, -1],
                }
                self.buffer.add_overall_stats(symmetry_stats)
        return pose_and_scale

    def get_object_rotation(self, graph_id, get_reverse_r=False):
        """Get the rotation of an object from the possible poses if resolved.

        This first checks whether we have recognized a unique pose of the object
        or if a symmetry is detected. If one of the two is true it returns the unique
        rotation(s), otherwise returns None.

        Args:
            graph_id: The object to check poses for.
            get_reverse_r: Whether to get the rotation that turns the model such
                that it would produce the sensed_displacements (False) or the rotation
                needed to turn the sensed_displacements into the model displacements.

        Returns:
            The rotation of the object if we know it.
        """
        unique_locations = self._get_possible_recent_paths(graph_id)
        location_is_unique = len(unique_locations) == 1
        all_poses = self.possible_poses[graph_id]
        euler_poses, unique_poses = get_unique_rotations(
            all_poses, self.pose_similarity_threshold, get_reverse_r
        )
        rotation_is_unique = len(unique_poses) == 1
        symmetry_detected = self._check_for_symmetry(
            np.array(euler_poses), len(unique_locations)
        )

        assert not (location_is_unique and rotation_is_unique and symmetry_detected)

        if (location_is_unique and rotation_is_unique) or symmetry_detected:
            return euler_poses, unique_poses
        else:
            self.last_unique_poses = np.array(euler_poses)
            self.last_num_unique_locations = len(unique_locations)
            return None, None

    def _check_for_symmetry(self, current_unique_poses, num_unique_locations):
        """Check for symmetry and update symmetry evidence count.

        Check if the last possible poses are the same as the current ones. This is
        taken as evidence for a symmetry in the object (poses are consistent with
        n successive observations).

        Returns:
            Whether symmetry was detected.
        """
        if self.last_unique_poses is None:
            return False  # need more steps to meet symmetry condition

        # Check if number of unique locations and poses has changed since the last step
        if (num_unique_locations == self.last_num_unique_locations) and (
            len(current_unique_poses) == len(self.last_unique_poses)
        ):
            # Check if the possible rotations are still the same
            equals = np.equal(current_unique_poses, self.last_unique_poses)
            if np.hstack(equals).all():
                self.symmetry_evidence += 1
            else:  # has to be consequtive
                self.symmetry_evidence = 0
        else:  # has to be consequtive
            self.symmetry_evidence = 0
        if self._enough_symmetry_evidence_accumulated():
            logger.info(f"Symmetry detected for poses {current_unique_poses}")
            return True
        else:
            return False

    def _enough_symmetry_evidence_accumulated(self):
        """Check if enough evidence for symmetry has been accumulated.

        Returns:
            Whether enough evidence for symmetry has been accumulated.
        """
        return self.symmetry_evidence >= self.required_symmetry_evidence

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _update_possible_matches(self, query):
        """Go through all objects and update possible matches.

        Args:
            query: current features at location.
        """
        consistent_objects = {}
        for graph_id in self.possible_matches:
            consistent = self._update_matches_using_features(
                query[0], query[1], graph_id
            )
            consistent_objects[graph_id] = consistent
        self._remove_inconsistent_objects(consistent_objects)

    def _update_matches_using_features(self, features, displacement, graph_id):
        """Use displacement to compare obseved features to possible graph features.

        At first observation (no displacement yet):
            Check which nodes in the graph are consistent with the observed features.
            -> these will be the possible start locations.
            Initialize possible poses for each location. Either by taking hard coded
            poses in 45 degree increments or by using the pose dependent features to
            determine possible poses for each location.
        For each following step:
            Get list of nodes that match with the observed features.
            For each possible location (path):
                For each possible pose at this location:
                    - Take the displacement and rotate it by the pose.
                    - Search location = location + rotated displacement
                    - Find nearest node to search location in the list of matching
                      feature nodes.
                    - check if pose dependent features at this node match with tested
                      pose. If not, look at next closest node (if distance <
                      max_node_distance)
                    - If we find a nearby matching node, add the search location as a
                      new possible location to the path (and the pose to possible
                      poses for this path). If not, remove this pose from possible
                      poses. If no poses are left for this path, remove the path.
        return len(possible_paths) > 0

        Args:
            features: Observed features at current time step.
            displacement: Displacement from previous location to current.
            graph_id: ID of model that should be tested.

        Returns:
            Whether we still have possible locations on this object.
        """
        new_possible_paths = []
        if displacement is None:
            # This is the first observation before we moved -> check where in the
            # graph the feature can be found
            (
                path_start_ids,
                new_possible_paths,
            ) = self.graph_memory.get_nodes_with_matching_features(
                graph_id=graph_id,
                features=features,
                list_of_lists=True,
            )

            if self.initial_possible_poses is None:
                # Get initial poses informed by pose features
                n_removed = 0
                for path_id, node_id in enumerate(path_start_ids):
                    possible_poses_for_path = self._get_informed_possible_poses(
                        graph_id, node_id, features
                    )
                    if len(possible_poses_for_path) > 0:
                        self.possible_poses[graph_id].append(possible_poses_for_path)
                    else:
                        new_possible_paths.pop(path_id - n_removed)
                        n_removed += 1
            else:
                # use uniformly distributed initial poses (in 45 degree intervals)
                self.possible_poses[graph_id] = [
                    self.initial_possible_poses.copy()
                    for _ in range(len(new_possible_paths))
                ]
        else:
            # We have already moved -> guide the node matching with displacement
            (
                new_possible_paths,
                new_possible_poses,
            ) = self._get_new_possible_paths_and_poses(graph_id, features, displacement)

            self.possible_poses[graph_id] = new_possible_poses
            if len(self.possible_poses[graph_id]) < 10:
                logger.info(
                    f"possible poses after matching for "
                    f"{graph_id}: {self.get_possible_poses()[graph_id]}"
                )
        self.possible_paths[graph_id] = new_possible_paths
        logger.debug(
            f"possible paths after matching for "
            f"{graph_id}: {len(self.possible_paths[graph_id])}"
        )
        return len(self.possible_paths[graph_id]) > 0

    def _get_new_possible_paths_and_poses(self, graph_id, features, displacement):
        """Use new displacement and features to update possible paths and poses.

        Returns:
            New possible paths and poses.
        """
        first_input_channel = list(features.keys())[0]
        displacement = displacement[first_input_channel]
        new_possible_paths = []
        new_possible_poses = []

        (
            feature_matched_node_ids,
            feature_matched_locs,
        ) = self.graph_memory.get_nodes_with_matching_features(
            graph_id=graph_id,
            features=features,
        )

        # if no points have the right features, it can't be this object
        if len(feature_matched_node_ids) == 0:
            return [], []

        # create a new KDtree with only eligible nodes
        reduced_tree = KDTree(feature_matched_locs, leaf_size=2)

        for path_id, path in enumerate(self.possible_paths[graph_id]):
            node_pos = path[-1]

            for pose in self.possible_poses[graph_id][path_id]:
                # This will just be one after the first step.
                search_pos = node_pos + pose.apply(displacement.copy())

                searching_near_nodes = True
                num_loops = 0
                closest_node_ds, closest_reduced_node_ids = reduced_tree.query(
                    [search_pos],
                    k=len(feature_matched_node_ids),
                    sort_results=True,
                )
                while searching_near_nodes and num_loops < len(
                    feature_matched_node_ids
                ):
                    # Find closest node using KD Tree search
                    closest_node_id = feature_matched_node_ids[
                        closest_reduced_node_ids[0][num_loops]
                    ]
                    closest_node_d = closest_node_ds[0][num_loops]

                    if closest_node_d > self.max_match_distance:
                        searching_near_nodes = False
                    else:
                        # Check if the feature pose matches the tested pose
                        new_pos_features = self.graph_memory.get_features_at_node(
                            graph_id,
                            first_input_channel,
                            closest_node_id,
                            feature_keys=["pose_vectors", "pose_fully_defined"],
                        )
                        pose_transformed_features = rotate_pose_dependent_features(
                            features[first_input_channel], pose
                        )
                        pose_features_match = self._match_pose_dependent_features(
                            pose_transformed_features,
                            new_pos_features,
                            first_input_channel,
                        )
                        if pose_features_match:
                            searching_near_nodes = False
                            new_possible_paths.append(
                                np.append(path, [search_pos], axis=0)
                            )
                            new_possible_poses.append([pose])
                        else:
                            num_loops += 1
        return new_possible_paths, new_possible_poses

    def _match_pose_dependent_features(
        self, query_features, node_features, input_channel
    ):
        """Determine whether pose features match.

        Compares the angle between observed and stored pose_vectors (from SM this
        corresponds to surface normal and curvature direction) and checks whether it is
        below the specified tolerance.

        Args:
            query_features: Observed features.
            node_features: Features at node that is being tested.
            input_channel: ?

        Returns:
            Whether feature matches given self.tolerances
        """
        vectors_to_check = 2
        if not query_features["pose_fully_defined"]:
            vectors_to_check = 1
        node_pose_vecs = np.array(node_features["pose_vectors"]).reshape((3, 3))
        for vec_id in range(vectors_to_check):
            angle = get_angle(
                query_features["pose_vectors"][vec_id],
                node_pose_vecs[vec_id],
            )
            if vec_id > 0:
                # account for the fact the curvature directions can be flipped
                # by 180 degrees
                # TODO H: what to do at higher level LMs?
                angle = np.pi / 2 - np.abs(angle - np.pi / 2)
            consistent = angle < self.tolerances[input_channel]["pose_vectors"][vec_id]
            if not consistent:
                return False
        return True

    def _remove_object_from_matches(self, graph_id):
        """Remove object and its poses from possible matches."""
        self.possible_matches.pop(graph_id)
        self.possible_poses[graph_id] = []
        self.possible_paths[graph_id] = []

    def _remove_inconsistent_objects(self, consistent_objects):
        """Remove objects from the list of possible objects.

        Args:
            consistent_objects: For each object whether it is still consistent.
        """
        for graph_id in consistent_objects.keys():
            if consistent_objects[graph_id] is False:
                self._remove_object_from_matches(graph_id)

    # ------------------------ Helper --------------------------

    def _get_possible_recent_paths(self, object_id, n_back=4):
        """Return n_back steps of the current possible unique paths.

        sometimes it happens that two paths and up on the same trajectory
        (I think because of matching to nodes that are nearby and not exactly
        at the current location). This deals with that because otherwise we
        never reach the stopping condition.

        Args:
            object_id: Object ID for which to return the paths.
            n_back: Number of recent locations to return.

        Returns:
            List of possible, unique, recent paths
        """
        possible_paths = self.get_possible_paths()[object_id]
        if isinstance(possible_paths[0], torch.Tensor):
            possible_paths = [path.clone().numpy() for path in possible_paths]

        if len(np.array(possible_paths).shape) == 1:
            unique_recent_paths = np.array(possible_paths)
        else:
            if np.array(possible_paths).shape[1] <= n_back:
                n_back = 0
            recent_paths = np.array(possible_paths)[:, -n_back:]
            unique_recent_paths = get_unique_paths(
                recent_paths, threshold=self.path_similarity_threshold
            )
        return unique_recent_paths

    def _get_informed_possible_poses(
        self,
        graph_id,
        node_id,
        sensed_features,
        n_samples=0,
        kappa=100,
    ):
        """Use the 1st input channel to get possible initial poses.

        Returns:
            Possible initial poses.
        """
        possible_poses = []
        all_input_channels = list(sensed_features.keys())
        first_input_channel = all_input_channels[0]
        node_directions = self.graph_memory.get_rotation_features_at_node(
            graph_id, node_id, first_input_channel
        )
        sensed_directions = sensed_features[first_input_channel]["pose_vectors"]
        # Check if PCs in patch are similar -> need to sample more directions
        if sensed_features[first_input_channel]["pose_fully_defined"]:
            # 2 possibilities since the curvature directions may be flipped
            possible_s_d = possible_sensed_directions(sensed_directions, 2)
        else:
            logger.debug(
                "PC 1 is similar to PC2 -> Their directions are not meaningful"
            )
            possible_s_d = possible_sensed_directions(
                sensed_directions, self.umbilical_num_poses
            )

        for s_d in possible_s_d:
            # Since we have orthonormal vectors and know their correspondence we can
            # directly calculate the rotation instead of using the Kabsch esimate used
            # in Rotation.align_vectors
            r, err = align_orthonormal_vectors(node_directions, s_d)
            if err < 1:
                possible_poses.append(r)
                for _ in range(n_samples):
                    # If we do this we need a better terminal condition for similar
                    # rotations or more robustness. n_sample currently set to 0.
                    rand_rot = self.rng.vonmises(0, kappa, 3)
                    rot = Rotation.from_euler(
                        "xyz", [rand_rot[0], rand_rot[1], rand_rot[2]]
                    )
                    r_sample = r * rot
                    possible_poses.append(r_sample)

        return possible_poses

    # ----------------------- Logging --------------------------

    def _add_detailed_stats(self, stats):
        stats["possible_paths"] = self.get_possible_paths()
        stats["possible_poses"] = self.get_possible_poses()
        stats["symmetry_evidence"] = self.symmetry_evidence
        return stats


class FeatureGraphMemory(GraphMemory):
    """Graph memory that matches objects by using features at locations."""

    def __init__(
        self,
        graph_delta_thresholds,
    ):
        """Initialize Graph Memory.

        Args:
            graph_delta_thresholds: Thresholds used to compare nodes in the graphs
                being learned, and thereby whether to include a new point or not.
        """
        super().__init__(graph_delta_thresholds=graph_delta_thresholds)

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        possible_matches = self.get_all_models_in_memory()
        possible_paths = {}
        possible_poses = {}
        # reset possible matches for paths on objects
        for graph_id in self.get_memory_ids():
            first_input_channel = self.get_input_channels_in_graph(graph_id)[0]
            # Get node IDs (fist element in x)
            possible_paths[graph_id] = self.get_graph_node_ids(
                graph_id, first_input_channel
            )
            possible_poses[graph_id] = []

        return possible_matches, possible_paths, possible_poses

    def get_rotation_features_at_node(self, graph_id, node_id, channel):
        """Get the rotation features at a node in the graph.

        Returns:
            The rotation features at a node in the graph.
        """
        node_r_features = self.get_features_at_node(
            graph_id,
            channel,
            node_id,
            feature_keys=["pose_vectors"],
        )
        node_directions = node_r_features["pose_vectors"]
        node_directions = np.array(node_directions).reshape((3, 3))
        return node_directions

    def get_nodes_with_matching_features(
        self,
        graph_id,
        features,
        list_of_lists=False,
    ) -> tuple[list, list]:
        """Get only nodes with matching features.

        Get a reduced list of nodes that includes only nodes with features
        that match the features dict passed here

        Args:
            graph_id: The graph descriptor e.g. 'mug'
            features: The observed features to be matched
            list_of_lists: should each location in the list be embedded in its own list
                (useful for some downstream operations)
            Defaults to False.

        Returns:
            The reduced lists of ids / locs.
        """
        first_input_channel = list(features.keys())[0]
        all_node_ids = self.get_graph_node_ids(graph_id, first_input_channel)
        all_node_locs = self.get_graph(graph_id, first_input_channel).pos
        # Just use first input channel for now. Since FeatureLM doesn't work with
        # heterarchy this should be fine. May want to allow for multiple sensor inputs
        # but probably not worth the time atm if we don't use this LM much.
        possible_nodes_idx = self._match_all_node_features(
            features, first_input_channel, graph_id
        )

        if list_of_lists:
            loc_lists = [[loc.numpy()] for loc in all_node_locs[possible_nodes_idx]]
            return all_node_ids[possible_nodes_idx], loc_lists
        else:
            return (
                all_node_ids[possible_nodes_idx],
                all_node_locs[possible_nodes_idx],
            )

    # ------------------ Logging & Saving ----------------------

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------

    def _match_all_node_features(self, features, input_channel, graph_id) -> np.ndarray:
        """Match observed features to all nodes in the graph.

        Match a list of the currently observed object features to an array of
        all nodes in the graph. First creates a list of features and tolerances
        where the index in the list matches those from self.feature_array

        Then it generates max and min permissible values, and compares these to
        the feature values from the self.feature_array of the whole graph.

        Circular variables (hue) must be matched differently, so this also gets
        a list of which vars are circular and then matches them differently

        Args:
            features: The observed features to be matched
            input_channel: ?
            graph_id: The graph descriptor e.g. 'mug'

        Returns:
            Array, where True~graph nodes matching ALL features,
            False~graph nodes with any non-matching features
        """
        shape_to_use = self.feature_array[graph_id][input_channel].shape[1]
        feature_order = self.feature_order[graph_id][input_channel]
        # generate the lists of features, tolerances, and whether features are circular
        tolerance_list = np.zeros(shape_to_use) * np.nan
        feature_list = np.zeros(shape_to_use) * np.nan
        circular_var = np.zeros(shape_to_use, dtype=bool)
        start_idx = 0
        features = features[input_channel]
        for feature in feature_order:
            if feature in [
                "pose_vectors",
                "pose_fully_defined",
            ]:
                continue
            if hasattr(features[feature], "__len__"):
                feature_length = len(features[feature])
            else:
                feature_length = 1
            end_idx = start_idx + feature_length
            feature_list[start_idx:end_idx] = features[feature]
            tolerance_list[start_idx:end_idx] = self.features_to_use[input_channel][
                feature
            ]
            circular_var[start_idx:end_idx] = (
                [True, False, False] if feature == "hsv" else False
            )
            start_idx = end_idx

        # use these arrays to find the max and min value for each feature
        min_value, max_value = np.zeros_like(feature_list), np.zeros_like(feature_list)
        min_value[circular_var] = (
            feature_list[circular_var] - tolerance_list[circular_var]
        ) % 1
        max_value[circular_var] = (
            feature_list[circular_var] + tolerance_list[circular_var]
        ) % 1
        min_value[~circular_var] = (
            feature_list[~circular_var] - tolerance_list[~circular_var]
        )
        max_value[~circular_var] = (
            feature_list[~circular_var] + tolerance_list[~circular_var]
        )
        min_larger_max = min_value > max_value

        # use the max and min value to test whether each graph node matches each feature
        in_range = np.zeros_like(self.feature_array[graph_id][input_channel])
        in_range[:, min_larger_max] = (
            self.feature_array[graph_id][input_channel][:, min_larger_max]
            >= min_value[min_larger_max]
        ) + (
            self.feature_array[graph_id][input_channel][:, min_larger_max]
            <= max_value[min_larger_max]
        )
        in_range[:, ~min_larger_max] = (
            self.feature_array[graph_id][input_channel][:, ~min_larger_max]
            >= min_value[~min_larger_max]
        ) * (
            self.feature_array[graph_id][input_channel][:, ~min_larger_max]
            <= max_value[~min_larger_max]
        )
        return np.all(in_range, axis=1)

    # ------------------------ Helper --------------------------

    # ----------------------- Logging --------------------------
