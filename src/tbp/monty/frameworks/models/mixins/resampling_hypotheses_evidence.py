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
from typing import Any, Dict

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

        # sampling parameters
        self.hypotheses_count_multiplier = hypotheses_count_multiplier
        self.hypotheses_old_to_new_ratio = hypotheses_old_to_new_ratio
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

    def _update_evidence(self, features, displacements, graph_id):
        """Resamples hypotheses and updates existing evidence."""
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
            old_count, informed_count, _ = self._sample_count(
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

    def _sample_count(self, input_channel, features, graph_id):
        """Returns the number of needed hypotheses."""
        graph_num_points = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        ).shape[0]

        full_informed_count = (
            graph_num_points * 2
            if features["patch"]["pose_fully_defined"]
            else graph_num_points * 8
        )

        if input_channel not in self.channel_hypothesis_mapping[graph_id].channels:
            return 0, full_informed_count, 0

        channel_range = self.channel_hypothesis_mapping[graph_id].channel_range(
            input_channel
        )
        current = channel_range[1] - channel_range[0]

        # calculate the total number of hypotheses needed
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
            new_sampled = needed - current

        # calculate how many informed and offspring hypotheses needed
        new_informed, new_offspring = (
            new_sampled * (1 - self.hypotheses_informed_to_offspring_ratio),
            new_sampled * self.hypotheses_informed_to_offspring_ratio,
        )
        # needed informed hypotheses should not exceed the available informed hypotheses
        # if trying to sample more hypotheses, set the available count as ceiling
        if new_informed > full_informed_count:
            new_informed = full_informed_count
            new_offspring = new_sampled - full_informed_count

        return (
            int(old_maintained),
            int(new_informed),
            int(new_offspring),
        )

    def _sample_informed(self, features, graph_id, informed_count, input_channel):
        if informed_count == 0:
            selected_locations = np.zeros((0, 3))
            selected_rotations = np.zeros((0, 3, 3))
            selected_evidence = np.zeros(0)
        else:
            (
                initial_possible_channel_locations,
                initial_possible_channel_rotations,
                channel_evidence,
            ) = self._get_initial_hypothesis_space(features, graph_id, input_channel)

            # Get the indices of the top `informed_count` values in `channel_evidence`
            top_indices = np.argsort(channel_evidence)[
                -informed_count:
            ]  # Get top indices

            # Select the corresponding entries from the original arrays
            selected_locations = initial_possible_channel_locations[top_indices]
            selected_rotations = initial_possible_channel_rotations[top_indices]
            selected_evidence = channel_evidence[top_indices]

        return selected_locations, selected_rotations, selected_evidence

    def _sample_old(self, features, graph_id, old_count, input_channel):
        if old_count == 0:
            selected_locations = np.zeros((0, 3))
            selected_rotations = np.zeros((0, 3, 3))
            selected_evidence = np.zeros(0)
        else:
            selected_locations = self.possible_locations[graph_id][:old_count]
            selected_rotations = self.possible_poses[graph_id][:old_count]
            selected_evidence = self.evidence[graph_id][:old_count]

        return selected_locations, selected_rotations, selected_evidence

    def _displace_hypotheses(
        self,
        features,
        locations,
        rotations,
        evidence,
        displacement,
        graph_id,
        input_channel,
    ):
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
        graph_id,
        input_channel,
        new_loc_hypotheses,
        new_rot_hypotheses,
        new_evidence,
    ):
        if input_channel not in self.channel_hypothesis_mapping[graph_id].channels:
            self.possible_locations[graph_id] = np.array(new_loc_hypotheses)
            self.possible_poses[graph_id] = np.array(new_rot_hypotheses)
            self.evidence[graph_id] = np.array(new_evidence)

            self.channel_hypothesis_mapping[graph_id].add_channel(
                input_channel, len(new_evidence)
            )
        else:
            channel_start_ix, channel_end_ix = self.channel_hypothesis_mapping[
                graph_id
            ].channel_range(input_channel)
            channel_size = channel_end_ix - channel_start_ix

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
