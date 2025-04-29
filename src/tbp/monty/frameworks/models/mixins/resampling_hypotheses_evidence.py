# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.utils.evidence_matching import ChannelMapper
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    rotate_pose_dependent_features,
)


class ResamplingHypothesesEvidenceMixin:
    """Mixin that adds resampling capability to EvidenceGraph learning modules.

    Compatible with:
        - EvidenceGraphLM
    """

    def __init__(
        self,
        *args: object,
        sampling_parameters: Dict[str, float],
        hypotheses_count_multiplier=1.0,
        hypotheses_old_to_new_ratio=0.0,
        hypotheses_informed_to_offspring_ratio=0.0,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)

        # Controls the shrinking or growth of hypothesis space size
        self.hypotheses_count_multiplier = hypotheses_count_multiplier

        # Controls the ratio of old to newly sampled hypotheses
        self.hypotheses_old_to_new_ratio = hypotheses_old_to_new_ratio

        # Controls the ratio of new informed to new offspring hypotheses
        # TODO This is set to 0 to sample only informed, offspring hypotheses are
        # currently not supported.
        self.hypotheses_informed_to_offspring_ratio = (
            hypotheses_informed_to_offspring_ratio
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure the mixin is used only with compatible learning modules.

        Raises:
            TypeError: If the mixin is used with a non-compatible learning module.
        """
        super().__init_subclass__(**kwargs)
        if not any(issubclass(b, (EvidenceGraphLM)) for b in cls.__bases__):
            raise TypeError(
                "ResamplingHypothesesEvidenceMixin must be mixed in with a subclass of "
                f"EvidenceGraphLM, got {cls.__bases__}"
            )

    def _update_evidence(
        self,
        features: Dict[str, ...],
        displacements: Optional[Dict[str, ...]],
        graph_id: str,
    ) -> None:
        """Update evidence of hypotheses space with resampling.

        Updates existing hypotheses space by:
            1. Calculating sample count for old and informed hypotheses
            2. Sampling hypotheses for old and informed hypotheses types
            3. Displacing (and updating evidence of) old hypotheses using
                given displacements
            4. Concatenating all samples (old + new) to rebuild the hypothesis space

        This process is repeated for each input channel in the graph.

        Args:
            features (dict): input features
            displacements (dict or None): given displacements
            graph_id (str): identifier of the graph being updated
        """
        start_time = time.time()

        if graph_id not in self.channel_hypothesis_mapping:
            self.channel_hypothesis_mapping[graph_id] = ChannelMapper()

        # get all usable input channels
        input_channels_to_use = [
            ic
            for ic in features.keys()
            if ic in self.get_input_channels_in_graph(graph_id)
        ]

        for input_channel in input_channels_to_use:
            # === GET SAMPLE COUNT ===
            old_count, informed_count = self._sample_count(
                input_channel, features, graph_id
            )

            # === SAMPLE HYPOTHESES ===
            old_locations, old_rotations, old_evidence = self._sample_old(
                features, graph_id, old_count, input_channel
            )
            informed_locations, informed_rotations, informed_evidence = (
                self._sample_informed(features, graph_id, informed_count, input_channel)
            )

            # === DISPLACE HYPOTHESES ===
            if old_count > 0:
                old_locations, old_evidence = self._displace_hypotheses(
                    features,
                    old_locations,
                    old_rotations,
                    old_evidence,
                    displacements,
                    graph_id,
                    input_channel,
                )

            # === CONCATENATE HYPOTHESES ===
            channel_locations = np.vstack([old_locations, informed_locations])
            channel_rotations = np.vstack([old_rotations, informed_rotations])
            channel_evidence = np.hstack([old_evidence, informed_evidence])

            # === RE-BUILD HYPOTHESIS SPACE ===
            self._replace_hypotheses_in_hpspace(
                graph_id=graph_id,
                input_channel=input_channel,
                new_loc_hypotheses=channel_locations,
                new_rot_hypotheses=channel_rotations,
                new_evidence=channel_evidence,
            )

        end_time = time.time()
        assert not np.isnan(np.max(self.evidence[graph_id])), "evidence contains NaN."
        logging.debug(
            f"evidence update for {graph_id} took "
            f"{np.round(end_time - start_time, 2)} seconds."
            f" New max evidence: {np.round(np.max(self.evidence[graph_id]), 3)}"
        )

    def _sample_count(
        self, input_channel: str, features: Dict[str, ...], graph_id: str
    ) -> Tuple[int, int]:
        """Calculates the number of old and informed hypotheses needed.

        Args:
            input_channel (str): The channel for which to calculate hypothesis count.
            features (dict): Input features containing pose information.
            graph_id (str): Identifier of the graph being queried.

        Returns:
            Tuple[int, int]: A tuple containing the number of old and new hypotheses
                needed. Old hypotheses are maintained from existing ones while new
                hypotheses are fully informed by pose sensory information.

        Notes:
            This function takes into account the following ratios:
              - `hypotheses_count_multiplier`: multiplier for total count calculation.
              - `hypotheses_old_to_new_ratio`: ratio between old and new hypotheses
                to be sampled.
        """
        graph_num_points = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        ).shape[0]

        full_informed_count = (
            graph_num_points * 2
            if features["patch"]["pose_fully_defined"]
            else graph_num_points * 8
        )

        # if hypothesis space does not exist, we initialize with informed hypotheses
        if input_channel not in self.channel_hypothesis_mapping[graph_id].channels:
            return 0, full_informed_count

        # calculate the total number of hypotheses needed
        current = self.channel_hypothesis_mapping[graph_id].channel_size(input_channel)
        needed = current * self.hypotheses_count_multiplier

        # calculate how many old and new hypotheses needed
        old_maintained, new_sampled = (
            needed * (1 - self.hypotheses_old_to_new_ratio),
            needed * self.hypotheses_old_to_new_ratio,
        )
        # needed old hypotheses should not exceed the existing hypotheses
        # if trying to maintain more hypotheses, set the available count as ceiling
        if old_maintained > current:
            old_maintained = current
            new_informed = needed - current

        # needed informed hypotheses should not exceed the available informed hypotheses
        # if trying to sample more hypotheses, set the available count as ceiling
        if new_informed > full_informed_count:
            new_informed = full_informed_count

        return (
            int(old_maintained),
            int(new_informed),
        )

    def _sample_informed(
        self,
        features: Dict[str, ...],
        graph_id: str,
        informed_count: int,
        input_channel: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples the specified number of fully informed hypotheses.

        Args:
            features (dict): Input features.
            graph_id (str): Identifier of the graph being queried.
            informed_count (int): Number of fully informed hypotheses to sample.
            input_channel: The channel for which hypotheses are sampled.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                selected locations, rotations, and evidence data.

        """
        # Return empty arrays for no hypotheses to sample
        if informed_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

        # TODO override `_get_initial_hypothesis_space` to postpone the rotation
        # calculation until after the points have been sampled based on
        # `_calculate_feature_evidence_for_all_nodes`.
        (
            initial_possible_channel_locations,
            initial_possible_channel_rotations,
            channel_evidence,
        ) = self._get_initial_hypothesis_space(features, graph_id, input_channel)

        # Get the indices of the top `informed_count` values in `channel_evidence`
        top_indices = np.argsort(channel_evidence)[-informed_count:]  # Get top indices

        # Select the corresponding entries from the original arrays
        selected_locations = initial_possible_channel_locations[top_indices]
        selected_rotations = initial_possible_channel_rotations[top_indices]
        selected_evidence = channel_evidence[top_indices]

        return selected_locations, selected_rotations, selected_evidence

    def _sample_old(
        self,
        features: Dict[str, ...],
        graph_id: str,
        old_count: int,
        input_channel: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples the specified number of existing hypotheses.

        Args:
            features (dict): Input features.
            graph_id (str): Identifier of the graph being queried.
            old_count (int): Number of existing hypotheses to sample.
            input_channel: The channel for which hypotheses are sampled.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                selected locations, rotations, and evidence scores.

        Notes:
            Currently samples based on available count instead of evidence slope.
        """
        # Return empty arrays for no hypotheses to sample
        if old_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

        # TODO implement sampling based on evidence slope.
        selected_locations = self.possible_locations[graph_id][:old_count]
        selected_rotations = self.possible_poses[graph_id][:old_count]
        selected_evidence = self.evidence[graph_id][:old_count]

        return selected_locations, selected_rotations, selected_evidence

    def _displace_hypotheses(
        self,
        features: Dict[str, ...],
        locations: np.ndarray,
        rotations: np.ndarray,
        evidence: np.ndarray,
        displacement: Dict[str, ...],
        graph_id: str,
        input_channel: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates existing hypotheses by displacing them.

        Args:
            features (dict): Input features
            locations (np.ndarray): Hypothesized sensor locations for each hypothesis
            rotations (np.ndarray): Hypothesized object rotations for each hypothesis
            evidence (np.ndarray): Current evidence value for each hypothesis
            displacement (dict): Sensor displacements for input channels
            graph_id (str): The ID of the current graph
            input_channel (str): The channel involved in hypotheses updating.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated sensor locations and evidence values.
        """
        evidence_threshold = self._get_evidence_update_threshold(graph_id)

        rotated_displacements = rotations.dot(displacement[input_channel])
        search_locations = locations + rotated_displacements

        hyp_ids_to_test = np.where(evidence >= evidence_threshold)[0]
        num_hypotheses_to_test = hyp_ids_to_test.shape[0]
        if num_hypotheses_to_test > 0:
            new_evidence = self._calculate_evidence_for_new_locations(
                graph_id,
                input_channel,
                search_locations[hyp_ids_to_test],
                rotations[hyp_ids_to_test],
                features,
            )
            min_update = np.clip(np.min(new_evidence), 0, np.inf)
            evidence_to_add = np.ones_like(evidence) * min_update
            evidence_to_add[hyp_ids_to_test] = new_evidence
            evidence = (
                evidence * self.past_weight + evidence_to_add * self.present_weight
            )
        return search_locations, evidence

    def _calculate_evidence_for_new_locations(
        self,
        graph_id,
        input_channel,
        search_locations,
        rotations,
        features,
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
            rotations,
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
        node_distance_weights = self._get_node_distance_weights(
            custom_nearest_node_dists
        )
        mask = node_distance_weights <= 0

        new_pos_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            nearest_node_ids,
            feature_keys=["pose_vectors", "pose_fully_defined"],
        )
        radius_evidence = self._get_pose_evidence_matrix(
            pose_transformed_features,
            new_pos_features,
            input_channel,
            node_distance_weights,
        )
        radius_evidence[mask] = -1
        node_distance_weights[mask] = 1

        if self.use_features_for_matching[input_channel]:
            node_feature_evidence = self._calculate_feature_evidence_for_all_nodes(
                features, input_channel, graph_id
            )
            hypothesis_radius_feature_evidence = node_feature_evidence[nearest_node_ids]
            hypothesis_radius_feature_evidence[mask] = 0
            radius_evidence = (
                radius_evidence
                + hypothesis_radius_feature_evidence * self.feature_evidence_increment
            )
        location_evidence = np.max(
            radius_evidence,
            axis=1,
        )
        return location_evidence

    def _replace_hypotheses_in_hpspace(
        self,
        graph_id: str,
        input_channel: str,
        new_loc_hypotheses: np.ndarray,
        new_rot_hypotheses: np.ndarray,
        new_evidence: np.ndarray,
    ) -> None:
        """Updates the hypothesis space for a given input channel in a graph.

        This function replaces existing or adds new hypotheses to the possible
        locations, poses and evidence arrays based on the provided information.

        Args:
            graph_id (str): The ID of the current graph to update.
            input_channel (str): Channel's name involved in updating the space
            new_loc_hypotheses (np.ndarray): New sensor locations hypotheses
            new_rot_hypotheses (np.ndarray): New object poses hypotheses
            new_evidence (np.ndarray): New evidence values for the input channel
        """
        # add a new channel to the mapping if the hypotheses space doesn't exist
        if input_channel not in self.channel_hypothesis_mapping[graph_id].channels:
            self.possible_locations[graph_id] = np.array(new_loc_hypotheses)
            self.possible_poses[graph_id] = np.array(new_rot_hypotheses)
            self.evidence[graph_id] = np.array(new_evidence)

            self.channel_hypothesis_mapping[graph_id].add_channel(
                input_channel, len(new_evidence)
            )
            return

        channel_start_ix, channel_end_ix = self.channel_hypothesis_mapping[
            graph_id
        ].channel_range(input_channel)
        channel_size = self.channel_hypothesis_mapping[graph_id].channel_size(
            input_channel
        )

        self.possible_locations[graph_id] = np.concatenate(
            [
                self.possible_locations[graph_id][:channel_start_ix],
                np.array(new_loc_hypotheses),
                self.possible_locations[graph_id][channel_end_ix:],
            ]
        )

        self.possible_poses[graph_id] = np.concatenate(
            [
                self.possible_poses[graph_id][:channel_start_ix],
                np.array(new_rot_hypotheses),
                self.possible_poses[graph_id][channel_end_ix:],
            ]
        )

        self.evidence[graph_id] = np.concatenate(
            [
                self.evidence[graph_id][:channel_start_ix],
                np.array(new_evidence),
                self.evidence[graph_id][channel_end_ix:],
            ]
        )

        self.channel_hypothesis_mapping[graph_id].resize_channel_by(
            input_channel, len(new_evidence) - channel_size
        )
