# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching_memory import EvidenceGraphMemory
from tbp.monty.frameworks.utils.evidence_matching import ChannelMapper
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
    get_angles_for_all_hypotheses,
    get_more_directions_in_plane,
    rotate_pose_dependent_features,
)


class GraphLMProtocol(Protocol):
    def get_graph(self, model_id, input_channel=None) -> Any: ...

    def get_input_channels_in_graph(self, graph_id: str) -> List[str]: ...


class EvidenceLMProtocol(Protocol):
    channel_hypothesis_mapping: Dict[str, ChannelMapper]
    current_mlh: Dict[str, str | list | Rotation | float]
    evidence: Dict[str, np.ndarray]
    evidence_update_threshold: Any
    feature_evidence_increment: int
    feature_weights: Any
    graph_memory: EvidenceGraphMemory
    initial_possible_poses: list | Literal["uniform", "informed"] | None
    max_match_distance: float
    max_nneighbors: int
    past_weight: float
    possible_locations: Dict[str, np.ndarray]
    possible_poses: Dict[str, np.ndarray]
    present_weight: float
    tolerances: Any
    use_features_for_matching: Dict[str, bool]
    x_percent_threshold: int


class EvidenceUpdaterProtocol(Protocol):
    def update_evidence(
        self, features: Dict, displacements: Optional[Dict], graph_id: str
    ) -> None: ...


class DefaultEvidenceUpdater(
    EvidenceLMProtocol, GraphLMProtocol, EvidenceUpdaterProtocol
):
    def update_evidence(
        self, features: Dict, displacements: Optional[Dict], graph_id: str
    ) -> None:
        """Update evidence for poses of graph_id.

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

        Args:
            features (Dict): A dictionary of input features.
            displacements (Optional[Dict]): A dictionary of displacements.
            graph_id (str): The ID of the graph to update.

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
            channel_mapper = ChannelMapper()
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
                channel_mapper.add_channel(input_channel, len(channel_evidence))
            self.possible_locations[graph_id] = np.concatenate(
                initial_possible_locations, axis=0
            )
            self.possible_poses[graph_id] = np.concatenate(
                initial_possible_rotations, axis=0
            )
            self.evidence[graph_id] = (
                np.concatenate(initial_evidence, axis=0) * self.present_weight
            )
            self.channel_hypothesis_mapping[graph_id] = channel_mapper
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

            channel_mapper = self.channel_hypothesis_mapping[graph_id]
            for input_channel in input_channels_to_use:
                # If channel features are observed for the first time, initialize
                # hypotheses for them.
                if input_channel not in channel_mapper.channels:
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
                    # Get the IDs range in hypothesis space for this channel
                    channel_start, channel_end = channel_mapper.channel_range(
                        input_channel
                    )
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
            f"{np.round(end_time - start_time, 2)} seconds."
            f" New max evidence: {np.round(np.max(self.evidence[graph_id]), 3)}"
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
        channel_mapper = self.channel_hypothesis_mapping[graph_id]
        channel_mapper.add_channel(input_channel, len(new_loc_hypotheses))

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
        neighboring nodes. Everything is weighted by the nodes distance from the search
        location.
        If there are no nodes in the search radius (max_match_distance), evidence = -1.

        We do this for every incoming input channel and its features if they are stored
        in the graph and take the average over the evidence from all input channels.

        Returns:
            The location evidence.
        """
        logging.debug(
            f"Calculating evidence for {graph_id} using input from {input_channel}"
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

        logging.debug(f"Determining possible poses using input from {input_channel}")
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
            assert percentage >= 0 and percentage <= 100, (
                "Percentage must be between 0 and 100"
            )
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
            evidence = np.zeros(initial_possible_channel_rotations.shape[0])
        return (
            initial_possible_channel_locations,
            initial_possible_channel_rotations,
            evidence,
        )

    def _get_node_distance_weights(self, distances):
        node_distance_weights = (
            self.max_match_distance - distances
        ) / self.max_match_distance
        return node_distance_weights

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
        pose_evidence_weighted = np.zeros(evidences_shape)
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
