# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Literal, Tuple, Type

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
    FeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.features_for_matching.selector import (  # noqa: E501
    DefaultFeaturesForMatchingSelector,
    FeaturesForMatchingSelector,
)
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    ChannelHypotheses,
    Hypotheses,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_displacer import (
    DefaultHypothesesDisplacer,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    all_usable_input_channels,
)
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_initial_possible_poses,
    possible_sensed_directions,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
)


class ResamplingHypothesesUpdater:
    """Hypotheses updater that resamples hypotheses at every step.

    This updater enables updating of the hypothesis space by resampling and rebuilding
    the hypothesis space at every step. We resample hypotheses from the existing
    hypothesis space, as well as new hypotheses informed by the sensed pose.

    The resampling process is governed by two main parameters:
      - `hypotheses_count_multiplier`: scales the total number of hypotheses every step.
      - `hypotheses_existing_to_new_ratio`: controls the proportion of existing vs.
          informed hypotheses during resampling.

    To reproduce the behavior of `DefaultHypothesesUpdater` sampling a fixed number of
    hypotheses only at the beginning of the episode, you can set
    `hypotheses_count_multiplier=1.0` and `hypotheses_existing_to_new_ratio=0.0`.

    Note:
        It would be better to decouple the amount of hypotheses added from the amount
        deleted in each step. At the moment, this is decided by the
        `hypotheses_count_multiplier`. For example, when the multiplier is set to 1.0,
        the hypotheses sampled is equal to the hypotheses removed. We ideally can
        decouple theses then use a slope threshold to decide on which hypotheses to
        remove, and use prediction error or other heuristics to decide to how many
        hypotheses to resample.
    """

    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        feature_evidence_calculator: Type[FeatureEvidenceCalculator] = (
            DefaultFeatureEvidenceCalculator
        ),
        feature_evidence_increment: int = 1,
        features_for_matching_selector: Type[FeaturesForMatchingSelector] = (
            DefaultFeaturesForMatchingSelector
        ),
        hypotheses_count_multiplier: float = 1.0,
        hypotheses_existing_to_new_ratio: float = 0.1,
        initial_possible_poses: Literal["uniform", "informed"]
        | list[Rotation] = "informed",
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
        umbilical_num_poses: int = 8,
    ):
        """Initializes the ResamplingHypothesesUpdater.

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
            feature_evidence_calculator: Class to calculate feature evidence for all
                nodes. Defaults to the default calculator.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence as opposed to the present_weight). Defaults to 1.
            features_for_matching_selector: Class to
                select if features should be used for matching. Defaults to the default
                selector.
            hypotheses_count_multiplier: Scales the total number of hypotheses
                every step. Defaults to 1.0.
            hypotheses_existing_to_new_ratio: Controls the proportion of the
                existing vs. newly sampled hypotheses during resampling. Defaults to
                0.0.
            initial_possible_poses: Initial
                possible poses that should be tested for. Defaults to "informed".
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
            umbilical_num_poses: Number of sampled rotations in the direction of
                the plane perpendicular to the point normal. These are sampled at
                umbilical points (i.e., points where PC directions are undefined).
        """
        self.feature_evidence_calculator = feature_evidence_calculator
        self.feature_evidence_increment = feature_evidence_increment
        self.feature_weights = feature_weights
        self.features_for_matching_selector = features_for_matching_selector
        self.graph_memory = graph_memory
        self.initial_possible_poses = get_initial_possible_poses(initial_possible_poses)
        self.tolerances = tolerances
        self.umbilical_num_poses = umbilical_num_poses

        self.use_features_for_matching = self.features_for_matching_selector.select(
            feature_evidence_increment=self.feature_evidence_increment,
            feature_weights=self.feature_weights,
            tolerances=self.tolerances,
        )
        self.hypotheses_displacer = DefaultHypothesesDisplacer(
            feature_evidence_increment=self.feature_evidence_increment,
            feature_weights=self.feature_weights,
            graph_memory=self.graph_memory,
            max_match_distance=max_match_distance,
            max_nneighbors=max_nneighbors,
            past_weight=past_weight,
            present_weight=present_weight,
            tolerances=self.tolerances,
            use_features_for_matching=self.use_features_for_matching,
        )

        # Controls the shrinking or growth of hypothesis space size
        # Cannot be less than 0
        self.hypotheses_count_multiplier = max(0, hypotheses_count_multiplier)

        # Controls the ratio of existing to newly sampled hypotheses
        # Bounded between 0 and 1
        self.hypotheses_existing_to_new_ratio = max(
            0, min(hypotheses_existing_to_new_ratio, 1)
        )

        # Dictionary of slope trackers, one for each graph_id
        self.evidence_slope_trackers: dict[str, EvidenceSlopeTracker] = {}

    def update_hypotheses(
        self,
        hypotheses: Hypotheses,
        features: dict,
        displacements: dict | None,
        graph_id: str,
        mapper: ChannelMapper,
        evidence_update_threshold: float,
    ) -> list[ChannelHypotheses]:
        """Update hypotheses based on sensor displacement and sensed features.

        Updates existing hypothesis space or initializes a new hypothesis space
        if one does not exist (i.e., at the beginning of the episode). Updating the
        hypothesis space includes displacing the hypotheses possible locations, as well
        as updating their evidence scores. This process is repeated for each input
        channel in the graph.

        Args:
            hypotheses: Hypotheses for all input channels in the graph_id
            features: Input features
            displacements: Given displacements
            graph_id: Identifier of the graph being updated
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel
            evidence_update_threshold: Evidence update threshold.

        Returns:
            The list of hypotheses updates to be applied to each input channel.
        """
        # Initialize a `EvidenceSlopeTracker` to keep track of evidence slopes
        # for hypotheses of a specific graph_id
        if graph_id not in self.evidence_slope_trackers:
            self.evidence_slope_trackers[graph_id] = EvidenceSlopeTracker()
        tracker = self.evidence_slope_trackers[graph_id]

        input_channels_to_use = all_usable_input_channels(
            features, self.graph_memory.get_input_channels_in_graph(graph_id)
        )

        hypotheses_updates = []

        for input_channel in input_channels_to_use:
            # Calculate sample count for each type
            existing_count, informed_count = self._sample_count(
                input_channel=input_channel,
                channel_features=features[input_channel],
                graph_id=graph_id,
                mapper=mapper,
                tracker=tracker,
            )

            # Sample hypotheses based on their type
            existing_hypotheses = self._sample_existing(
                existing_count=existing_count,
                hypotheses=hypotheses,
                input_channel=input_channel,
                mapper=mapper,
                tracker=tracker,
            )
            informed_hypotheses = self._sample_informed(
                channel_features=features[input_channel],
                graph_id=graph_id,
                informed_count=informed_count,
                input_channel=input_channel,
                tracker=tracker,
            )

            # We only displace existing hypotheses since the newly resampled hypotheses
            # should not be affected by the displacement from the last sensory input.
            if existing_count > 0:
                existing_hypotheses = (
                    self.hypotheses_displacer.displace_hypotheses_and_compute_evidence(
                        channel_displacement=displacements[input_channel],
                        channel_features=features[input_channel],
                        evidence_update_threshold=evidence_update_threshold,
                        graph_id=graph_id,
                        possible_hypotheses=existing_hypotheses,
                        total_hypotheses_count=hypotheses.evidence.shape[0],
                    )
                )

            # Concatenate and rebuild channel hypotheses
            channel_hypotheses = ChannelHypotheses(
                input_channel=input_channel,
                locations=np.vstack(
                    [existing_hypotheses.locations, informed_hypotheses.locations]
                ),
                poses=np.vstack([existing_hypotheses.poses, informed_hypotheses.poses]),
                evidence=np.hstack(
                    [existing_hypotheses.evidence, informed_hypotheses.evidence]
                ),
            )
            hypotheses_updates.append(channel_hypotheses)

            # Update tracker evidence
            tracker.update(channel_hypotheses.evidence, input_channel)

        return hypotheses_updates

    def _num_hyps_per_node(self, channel_features: dict) -> int:
        """Calculate the number of hypotheses per node.

        Args:
            channel_features: Features for the input channel.

        Returns:
            The number of hypotheses per node.
        """
        if self.initial_possible_poses is None:
            return (
                2
                if channel_features["pose_fully_defined"]
                else self.umbilical_num_poses
            )
        else:
            return len(self.initial_possible_poses)

    def _sample_count(
        self,
        input_channel: str,
        channel_features: dict,
        graph_id: str,
        mapper: ChannelMapper,
        tracker: EvidenceSlopeTracker,
    ) -> Tuple[int, int]:
        """Calculates the number of existing and informed hypotheses needed.

        Args:
            input_channel: The channel for which to calculate hypothesis count.
            channel_features: Input channel features containing pose information.
            graph_id: Identifier of the graph being queried.
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            A tuple containing the number of existing and new hypotheses needed.
            Existing hypotheses are maintained from existing ones while new hypotheses
            will be initialized, informed by pose sensory information.

        Notes:
            This function takes into account the following ratios:
              - `hypotheses_count_multiplier`: multiplier for total count calculation.
              - `hypotheses_existing_to_new_ratio`: ratio between existing and new
                hypotheses to be sampled.
        """
        graph_num_points = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        ).shape[0]
        num_hyps_per_node = self._num_hyps_per_node(channel_features)
        full_informed_count = graph_num_points * num_hyps_per_node

        # If hypothesis space does not exist, we initialize with informed hypotheses
        if input_channel not in mapper.channels:
            return 0, full_informed_count

        # Calculate the total number of hypotheses needed
        current = mapper.channel_size(input_channel)
        needed = current * self.hypotheses_count_multiplier

        # Calculate how many existing and new hypotheses needed
        existing_maintained, new_informed = (
            needed * (1 - self.hypotheses_existing_to_new_ratio),
            needed * self.hypotheses_existing_to_new_ratio,
        )

        # Needed existing hypotheses should not exceed the existing hypotheses
        # if trying to maintain more hypotheses, set the available count as ceiling

        # We make sure that `new_informed` is divisible by the number of hypotheses
        # per graph node. This allows for sampling the graph nodes first (according
        # to evidence) then multiply by the `num_hyps_per_node`, as shown in
        # `_sample_informed`.
        if existing_maintained > current:
            existing_maintained = current
            new_informed = needed - current
            new_informed -= new_informed % num_hyps_per_node

        # Needed informed hypotheses should not exceed the available informed hypotheses
        # If trying to sample more hypotheses, set the available count as ceiling
        if new_informed > full_informed_count:
            new_informed = full_informed_count

        # Additional adjustment based on valid mask
        must_keep = int(np.sum(~tracker.removable_indices_mask(input_channel)))
        if must_keep > existing_maintained:
            existing_maintained = must_keep
            new_informed = needed - existing_maintained

        return (
            int(existing_maintained),
            int(new_informed),
        )

    def _sample_existing(
        self,
        existing_count: int,
        hypotheses: Hypotheses,
        input_channel: str,
        mapper: ChannelMapper,
        tracker: EvidenceSlopeTracker,
    ) -> ChannelHypotheses:
        """Samples the specified number of existing hypotheses to retain.

        Args:
            existing_count: Number of existing hypotheses to sample.
            hypotheses: Hypotheses for all input channels in the graph_id.
            input_channel: The channel for which to sample existing hypotheses.
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            The sampled existing hypotheses.
        """
        # Return empty arrays for no hypotheses to sample
        if existing_count == 0:
            # Clear all channel hypotheses from the tracker
            tracker.clear_hyp(input_channel)

            return ChannelHypotheses(
                input_channel=input_channel,
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
            )

        keep_ids, remove_ids = tracker.calculate_keep_and_remove_ids(
            num_keep=existing_count,
            channel=input_channel,
        )

        # Update tracker by removing the remove_ids
        tracker.remove_hyp(remove_ids, input_channel)

        channel_hypotheses = mapper.extract_hypotheses(hypotheses, input_channel)
        return ChannelHypotheses(
            input_channel=channel_hypotheses.input_channel,
            locations=channel_hypotheses.locations[keep_ids],
            poses=channel_hypotheses.poses[keep_ids],
            evidence=channel_hypotheses.evidence[keep_ids],
        )

    def _sample_informed(
        self,
        channel_features: dict,
        informed_count: int,
        graph_id: str,
        input_channel: str,
        tracker: EvidenceSlopeTracker,
    ) -> ChannelHypotheses:
        """Samples the specified number of fully informed hypotheses.

        This method selects hypotheses that are most likely to be informative based on
        feature evidence. Specifically, it identifies the top-k nodes with the highest
        evidence scores and samples hypotheses only from those nodes, making the process
        more efficient than uniformly sampling from all graph nodes.

        The sampling includes:
          - Selecting the top-k node indices based on evidence scores, where k is
             determined by the `informed_count` and the number of hypotheses per node.
          - Fetching the 3D locations of only the selected top-k nodes.
          - Generating rotations for each hypothesis using one of two strategies:
            (a) If `initial_possible_poses` is set, rotations are uniformly sampled or
                user-defined.
            (b) Otherwise, alignments are computed between stored node poses and
                sensed directions.

        This targeted sampling improves efficiency by avoiding unnecessary computation
        for nodes with low evidence, especially beneficial when informed sampling occurs
        at every step.

        Args:
            channel_features: Input channel features.
            informed_count: Number of fully informed hypotheses to sample.
            graph_id: Identifier of the graph being queried.
            input_channel: The channel for which to sample informed hypotheses.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            The sampled informed hypotheses.

        """
        # Return empty arrays for no hypotheses to sample
        if informed_count == 0:
            return ChannelHypotheses(
                input_channel=input_channel,
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
            )

        num_hyps_per_node = self._num_hyps_per_node(channel_features)
        # === Calculate selected evidence by top-k indices === #
        if self.use_features_for_matching[input_channel]:
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
            # Find the indices for the nodes with highest evidence scores. The sorting
            # is done in ascending order, so extract the indices from the end of
            # the argsort array. We get the needed number of informed nodes not
            # the number of needed hypotheses.
            top_indices = np.argsort(node_feature_evidence)[
                -int(informed_count // num_hyps_per_node) :
            ]
            node_feature_evidence_filtered = (
                node_feature_evidence[top_indices] * self.feature_evidence_increment
            )
        else:
            num_nodes = self.graph_memory.get_num_nodes_in_graph(graph_id)
            top_indices = np.arange(num_nodes)[
                : int(informed_count // num_hyps_per_node)
            ]
            node_feature_evidence_filtered = np.zeros(len(top_indices))

        selected_feature_evidence = np.tile(
            node_feature_evidence_filtered, num_hyps_per_node
        )

        # === Calculate selected locations by top-k indices === #
        all_channel_locations_filtered = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        )[top_indices]
        selected_locations = np.tile(
            all_channel_locations_filtered, (num_hyps_per_node, 1)
        )

        # === Calculate selected rotations by top-k indices === #
        if self.initial_possible_poses is None:
            node_directions_filtered = (
                self.graph_memory.get_rotation_features_at_all_nodes(
                    graph_id, input_channel
                )[top_indices]
            )
            sensed_directions = channel_features["pose_vectors"]
            possible_s_d = possible_sensed_directions(
                sensed_directions, num_hyps_per_node
            )
            selected_rotations = np.vstack(
                [
                    align_multiple_orthonormal_vectors(
                        node_directions_filtered, s_d, as_scipy=False
                    )
                    for s_d in possible_s_d
                ]
            )

        else:
            selected_rotations = np.vstack(
                [
                    np.tile(
                        rotation.as_matrix()[np.newaxis, ...],
                        (len(top_indices), 1, 1),
                    )
                    for rotation in self.initial_possible_poses
                ]
            )

        # Add hypotheses to slope trackers
        tracker.add_hyp(selected_feature_evidence.shape[0], input_channel)

        return ChannelHypotheses(
            input_channel=input_channel,
            locations=selected_locations,
            poses=selected_rotations,
            evidence=selected_feature_evidence,
        )
