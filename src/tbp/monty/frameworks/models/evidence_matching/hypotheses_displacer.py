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
from dataclasses import dataclass
from typing import Protocol, Type

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
    FeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import ChannelHypotheses
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_angles_for_all_hypotheses,
    rotate_pose_dependent_features,
)

logger = logging.getLogger(__name__)


@dataclass
class HypothesisDisplacerTelemetry:
    mlh_prediction_error: float | None


class HypothesesDisplacer(Protocol):
    def displace_hypotheses_and_compute_evidence(
        self,
        channel_displacement: np.ndarray,
        channel_features: dict,
        evidence_update_threshold: float,
        graph_id: str,
        possible_hypotheses: ChannelHypotheses,
        total_hypotheses_count: int,
    ) -> ChannelHypotheses:
        """Updates evidence by comparing features after applying sensed displacement.

        This function applies the sensor displacement to the existing hypothesis and
        uses the result as search locations for comparing the sensed features. This
        comparison is used to update the evidence scores of the existing hypotheses. The
        hypotheses locations are updated to the new locations (i.e., after displacement)

        Args:
            channel_displacement: Channel-specific sensor displacement.
            channel_features: Channel-specific input features.
            evidence_update_threshold: Evidence update threshold.
            graph_id: The ID of the current graph
            possible_hypotheses: Channel-specific possible
                hypotheses.
            total_hypotheses_count: Total number of hypotheses in the graph.

        Returns:
            Displaced hypotheses with computed evidence.
        """
        ...


class DefaultHypothesesDisplacer:
    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        use_features_for_matching: dict[str, bool],
        feature_evidence_calculator: Type[
            FeatureEvidenceCalculator
        ] = DefaultFeatureEvidenceCalculator,
        feature_evidence_increment: int = 1,
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
    ):
        """Initializes the DefaultHypothesesDisplacer.

        Args:
            feature_weights: How much should each feature be weighted when
                calculating the evidence update for hypothesis. Weights are stored in a
                dictionary with keys corresponding to features (same as keys in
                tolerances).
            graph_memory: The graph memory to read graphs from.
            max_match_distance: Maximum distance of a tested and stored location
                to be matched.
            tolerances: How much can each observed feature deviate from the
                stored features to still be considered a match.
            use_features_for_matching: Dictionary mapping input channels to
                booleans indicating whether to use features for matching.
            feature_evidence_calculator: Class to calculate feature evidence for all
                nodes. Defaults to the default calculator.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence as opposed to the present_weight). Defaults to 1.
            max_nneighbors: Maximum number of nearest neighbors to consider in the
                radius of a hypothesis for calculating the evidence. Defaults to 3.
            past_weight: How much should the evidence accumulated so far be
                weighted when combined with the evidence from the most recent
                observation. Defaults to 1.
            present_weight: How much should the current evidence be weighted
                when added to the previous evidence. If past_weight and present_weight
                add up to 1, the evidence is bounded and can't grow infinitely. Defaults
                to 1.
                NOTE: right now this doesn't give as good performance as with unbounded
                evidence since we don't keep a full history of what we saw. With a more
                efficient policy and better parameters that may be possible to use
                though and could help when moving from one object to another and to
                generally make setting thresholds etc. more intuitive.
        """
        self.feature_evidence_calculator = feature_evidence_calculator
        self.feature_evidence_increment = feature_evidence_increment
        self.feature_weights = feature_weights
        self.graph_memory = graph_memory
        self.max_match_distance = max_match_distance
        self.max_nneighbors = max_nneighbors
        self.past_weight = past_weight
        self.present_weight = present_weight
        self.tolerances = tolerances
        self.use_features_for_matching = use_features_for_matching

    def displace_hypotheses_and_compute_evidence(
        self,
        channel_displacement: np.ndarray,
        channel_features: dict,
        evidence_update_threshold: float,
        graph_id: str,
        possible_hypotheses: ChannelHypotheses,
        total_hypotheses_count: int,
    ) -> tuple[ChannelHypotheses, HypothesisDisplacerTelemetry]:
        # Have to do this for all hypotheses so we don't loose the path information
        rotated_displacements = possible_hypotheses.poses.dot(channel_displacement)
        search_locations = possible_hypotheses.locations + rotated_displacements

        # Get indices of hypotheses with evidence > threshold
        hyp_ids_to_test = np.where(
            possible_hypotheses.evidence >= evidence_update_threshold
        )[0]
        num_hypotheses_to_test = hyp_ids_to_test.shape[0]
        if num_hypotheses_to_test > 0:
            logger.info(
                f"Testing {num_hypotheses_to_test} out of "
                f"{total_hypotheses_count} hypotheses for {graph_id} "
                f"(evidence > {evidence_update_threshold})"
            )

            # Get evidence update for all hypotheses with evidence > current
            # _evidence_update_threshold
            new_evidence = self._calculate_evidence_for_new_locations(
                graph_id=graph_id,
                input_channel=possible_hypotheses.input_channel,
                search_locations=search_locations[hyp_ids_to_test],
                channel_possible_poses=possible_hypotheses.poses[hyp_ids_to_test],
                channel_features=channel_features,
            )
            min_update = np.clip(np.min(new_evidence), 0, np.inf)

            # Alternatives (no update to other Hs or adding avg) left in
            # here in case we want to revert back to those.
            # avg_update = np.mean(new_evidence)
            # evidence_to_add = np.zeros_like(channel_hypotheses_evidence)
            evidence_to_add = np.ones_like(possible_hypotheses.evidence) * min_update
            evidence_to_add[hyp_ids_to_test] = new_evidence

            mlh_index = np.argmax(possible_hypotheses.evidence)
            evidence_for_mlh = evidence_to_add[mlh_index]
            # Mapping evidence values from range [-1, 2] to [0, 3], then dividing by 3
            # to get a value in range [0, 1].
            mlh_prediction_error = (-evidence_for_mlh + 2) / 3

            # If past and present weight add up to 1, equivalent to
            # np.average and evidence will be bound to [-1, 2]. Otherwise it
            # keeps growing.
            evidence = (
                possible_hypotheses.evidence * self.past_weight
                + evidence_to_add * self.present_weight
            )
        else:
            evidence = possible_hypotheses.evidence
            # If we haven't moved yet, there is no prediction, and thus no error
            mlh_prediction_error = None

        return ChannelHypotheses(
            input_channel=possible_hypotheses.input_channel,
            evidence=evidence,
            locations=search_locations,
            poses=possible_hypotheses.poses,
        ), HypothesisDisplacerTelemetry(mlh_prediction_error=mlh_prediction_error)

    def _calculate_evidence_for_new_locations(
        self,
        graph_id: str,
        input_channel: str,
        search_locations: np.ndarray,
        channel_possible_poses: np.ndarray,
        channel_features: dict,
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
        logger.debug(
            f"Calculating evidence for {graph_id} using input from {input_channel}"
        )

        pose_transformed_features = rotate_pose_dependent_features(
            channel_features,
            channel_possible_poses,
        )
        # Get max_nneighbors nearest nodes to search locations.
        nearest_node_ids = self.graph_memory.get_graph(
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
        max_abs_curvature = get_relevant_curvature(channel_features)
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

        # If no feature weights are provided besides the ones for surface_normal
        # and curvature_directions we don't need to calculate feature evidence.
        if self.use_features_for_matching[input_channel]:
            # add evidence if features match
            node_feature_evidence = self.feature_evidence_calculator.calculate(
                channel_feature_array=self.graph_memory.get_feature_array(graph_id)[
                    input_channel
                ],
                channel_feature_order=self.graph_memory.get_feature_order(graph_id)[
                    input_channel
                ],
                channel_feature_weights=self.feature_weights[input_channel],
                channel_query_features=channel_features,
                channel_tolerances=self.tolerances[input_channel],
                input_channel=input_channel,
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
        surface_normal_error = get_angles_for_all_hypotheses(
            # shape of node_features[input_channel]["pose_vectors"]: (nH, knn, 9)
            node_features["pose_vectors"][:, :, :3],
            query_features["pose_vectors"][:, 0],  # shape (nH, 3)
        )
        # Divide error by 2 so it is in range [0, pi/2]
        # Apply sin -> [0, 1]. Subtract 0.5 -> [-0.5, 0.5]
        # Negate the error to get evidence (lower error is higher evidence)
        surface_normal_evidence = -(np.sin(surface_normal_error / 2) - 0.5)
        surface_normal_weight = self.feature_weights[input_channel]["pose_vectors"][0]
        # If curvatures are same the directions are meaningless
        #  -> set curvature angle error to zero.
        if not query_features["pose_fully_defined"]:
            cd1_weight = 0
            # Only calculate curv dir angle if sensed curv dirs are meaningful
            cd1_evidence = np.zeros(surface_normal_error.shape)
            # TODO: Test whether we should double the SN evidence if no
            # curvatures are sensed and pose_fully_defined == False at node.
            # i.e. move use_cd from else block and set
            # surface_normal_evidence[np.logical_not(use_cd)] *= 2 (see PR#446)
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
            # We then apply the same operations as on surface_normal error to get
            # cd1_evidence
            # in range [-0.5, 0.5]
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            # nodes where pc1==pc2 receive no cd evidence but twice the surface_normal
            # evidence
            # -> overall evidence can be in range [-1, 1]
            cd1_evidence = cd1_evidence * use_cd
        # weight angle errors by feature weights
        # if sensed pc1==pc2 cd1_weight==0 and overall evidence is in [-0.5, 0.5]
        # otherwise it is in [-1, 1].
        pose_evidence_weighted += (
            surface_normal_evidence * surface_normal_weight + cd1_evidence * cd1_weight
        )
        return pose_evidence_weighted
