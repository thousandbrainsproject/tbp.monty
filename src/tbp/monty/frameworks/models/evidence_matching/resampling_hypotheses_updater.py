# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Tuple, Type

import numpy as np
import numpy.typing as npt
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
    HypothesesUpdateTelemetry,
    all_usable_input_channels,
)
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
    HypothesesSelection,
    InvalidEvidenceThresholdConfig,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_initial_possible_poses,
    possible_sensed_directions,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
)


@dataclass
class ChannelHypothesesResamplingTelemetry:
    """Hypotheses resampling telemetry for a channel.

    For a given input channel, this class stores which hypotheses were removed or
    added at the current step.

    Note:
        TODO: Additional description here regarding availability of hypotheses
        identified by `removed_ids`.
    """

    added_ids: npt.NDArray[np.int_]
    ages: npt.NDArray[np.int_]
    evidence_slopes: npt.NDArray[np.float64]
    removed_ids: npt.NDArray[np.int_]


class ResamplingHypothesesUpdater:
    """Hypotheses updater that resamples hypotheses at every step.

    This updater enables updating of the hypothesis space by resampling and rebuilding
    the hypothesis space at every step. We resample hypotheses from the existing
    hypothesis space, as well as new hypotheses informed by the sensed pose.

    The resampling process is governed by two main parameters:
      - `resampling_multiplier`: Determines the number of the hypotheses to resample
        as a multiplier of the object graph nodes.
      - `evidence_slope_threshold`: Hypotheses below this threshold are deleted.

    To reproduce the behavior of `DefaultHypothesesUpdater` sampling a fixed number of
    hypotheses only at the beginning of the episode, you can set
    `resampling_multiplier=0.0` and `evidence_slope_threshold=-1.0`.
    """

    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        evidence_threshold_config: Literal["all"],
        feature_evidence_calculator: Type[FeatureEvidenceCalculator] = (
            DefaultFeatureEvidenceCalculator
        ),
        feature_evidence_increment: int = 1,
        features_for_matching_selector: Type[FeaturesForMatchingSelector] = (
            DefaultFeaturesForMatchingSelector
        ),
        resampling_multiplier: float = 0.1,
        evidence_slope_threshold: float = 0.0,
        include_telemetry: bool = False,
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
            evidence_threshold_config: How to decide which hypotheses
                should be updated. In the `ResamplingHypothesesUpdater` we always
                update 'all' hypotheses. Hypotheses with decreasing evidence are deleted
                instead of excluded from updating. Must be set to 'all'.
            feature_evidence_calculator: Class to calculate feature evidence for all
                nodes. Defaults to the default calculator.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence as opposed to the present_weight). Defaults to 1.
            features_for_matching_selector: Class to
                select if features should be used for matching. Defaults to the default
                selector.
            resampling_multiplier: Determines the number of the hypotheses to resample
                as a multiplier of the object graph nodes. Value of 0.0 results in no
                resampling. Value can be greater than 1 but not to exceed the
                `num_hyps_per_node` of the current step. Defaults to 0.1.
            evidence_slope_threshold: Hypotheses below this threshold are deleted.
                Defaults to 0.0.
            include_telemetry: Flag to control if we want to calculate and return the
                resampling telemetry in the `update_hypotheses` method. Defaults to
                False.
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
                the plane perpendicular to the surface normal. These are sampled at
                umbilical points (i.e., points where PC directions are undefined).

        Raises:
            ValueError: If the resampling_multiplier is less than 0
            InvalidEvidenceThresholdConfig: If `evidence_threshold_config` is not
                set to "all".

        """
        if evidence_threshold_config != "all":
            raise InvalidEvidenceThresholdConfig(
                "evidence_threshold_config must be "
                "'all' for `ResamplingHypothesesUpdater`"
            )

        self.feature_evidence_calculator = feature_evidence_calculator
        self.feature_evidence_increment = feature_evidence_increment
        self.feature_weights = feature_weights
        self.features_for_matching_selector = features_for_matching_selector
        self.resampling_multiplier = resampling_multiplier
        self.evidence_slope_threshold = evidence_slope_threshold
        self.graph_memory = graph_memory
        self.include_telemetry = include_telemetry
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

        # resampling multiplier should not be less than 0 (no resampling)
        if self.resampling_multiplier < 0:
            raise ValueError("resampling_multiplier should be >= 0")

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
    ) -> tuple[list[ChannelHypotheses], HypothesesUpdateTelemetry]:
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
            A tuple containing the list of hypotheses updates to be applied to each
            input channel and hypotheses update telemetry for analysis. The hypotheses
            update telemetry is a dictionary containing:
                - added_ids: IDs of hypotheses added during resampling at the current
                    timestep.
                - ages: The ages of the hypotheses as tracked by the
                    `EvidenceSlopeTracker`.
                - evidence_slopes: The slopes extracted from the `EvidenceSlopeTracker`.
                - removed_ids: IDs of hypotheses removed during resampling. Note that
                    these IDs can only be used to index hypotheses from the previous
                    timestep.
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
        resampling_telemetry: dict[str, Any] = {}

        for input_channel in input_channels_to_use:
            # Calculate sample count for each type
            hypotheses_selection, informed_count = self._sample_count(
                input_channel=input_channel,
                channel_features=features[input_channel],
                graph_id=graph_id,
                mapper=mapper,
                tracker=tracker,
            )

            # Sample hypotheses based on their type
            existing_hypotheses = self._sample_existing(
                hypotheses_selection=hypotheses_selection,
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
            if len(hypotheses_selection.maintain_ids) > 0:
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

            if self.include_telemetry:
                resampling_telemetry[input_channel] = asdict(
                    ChannelHypothesesResamplingTelemetry(
                        added_ids=(
                            np.arange(len(channel_hypotheses.evidence))[
                                -len(informed_hypotheses.evidence) :
                            ]
                            if len(informed_hypotheses.evidence) > 0
                            else np.array([], dtype=np.int_)
                        ),
                        ages=tracker.hyp_ages(input_channel),
                        evidence_slopes=tracker.calculate_slopes(input_channel),
                        removed_ids=hypotheses_selection.remove_ids,
                    )
                )

        return (
            hypotheses_updates,
            resampling_telemetry if self.include_telemetry else None,
        )

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
    ) -> Tuple[HypothesesSelection, int]:
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
            A tuple containing the hypotheses selection and count of new hypotheses
            needed. Hypotheses selection are maintained from existing ones while new
            hypotheses will be initialized, informed by pose sensory information.

        Notes:
            This function takes into account the following parameters:
              - `resampling_multiplier`: The number of hypotheses to resample. This
                is defined as a multiplier of the number of nodes in the object graph.
              - `evidence_slope_threshold`: This dictates how many hypotheses to
                delete. Hypotheses below this threshold are deleted.
        """
        graph_num_points = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        ).shape[0]
        num_hyps_per_node = self._num_hyps_per_node(channel_features)
        full_informed_count = graph_num_points * num_hyps_per_node

        # If hypothesis space does not exist, we initialize with informed hypotheses.
        # Should we remove this now that we are resampling? We can sample the
        # same number of hypotheses during initialization as in every other step.
        if input_channel not in mapper.channels:
            return HypothesesSelection(maintain_mask=[]), full_informed_count

        # This makes sure that we do not request more than the available number of
        # informed hypotheses
        resampling_multiplier = min(self.resampling_multiplier, num_hyps_per_node)

        # Calculate the total number of informed hypotheses to be resampled
        new_informed = round(graph_num_points * resampling_multiplier)
        new_informed -= new_informed % num_hyps_per_node

        # Returns a selection of hypotheses to maintain/delete
        hypotheses_selection = tracker.select_hypotheses(
            slope_threshold=self.evidence_slope_threshold, channel=input_channel
        )

        return (
            hypotheses_selection,
            new_informed,
        )

    def _sample_existing(
        self,
        hypotheses_selection: HypothesesSelection,
        hypotheses: Hypotheses,
        input_channel: str,
        mapper: ChannelMapper,
        tracker: EvidenceSlopeTracker,
    ) -> tuple[ChannelHypotheses, npt.NDArray[np.int_]]:
        """Samples the specified number of existing hypotheses to retain.

        Args:
            hypotheses_selection: The selection of hypotheses to maintain/remove.
            hypotheses: Hypotheses for all input channels in the graph_id.
            input_channel: The channel for which to sample existing hypotheses.
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            A tuple of sampled existing hypotheses and the IDs of the hypotheses to
            remove.
        """
        maintain_ids = hypotheses_selection.maintain_ids

        # Return empty arrays for no hypotheses to sample
        if len(maintain_ids) == 0:
            # Clear all channel hypotheses from the tracker
            remove_ids = np.arange(tracker.total_size(input_channel))
            tracker.clear_hyp(input_channel)

            channel_hypotheses = ChannelHypotheses(
                input_channel=input_channel,
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
            )
            return channel_hypotheses

        # Update tracker by removing the remove_ids
        tracker.remove_hyp(hypotheses_selection.remove_ids, input_channel)

        channel_hypotheses = mapper.extract_hypotheses(hypotheses, input_channel)
        maintained_channel_hypotheses = ChannelHypotheses(
            input_channel=channel_hypotheses.input_channel,
            locations=channel_hypotheses.locations[maintain_ids],
            poses=channel_hypotheses.poses[maintain_ids],
            evidence=channel_hypotheses.evidence[maintain_ids],
        )
        return maintained_channel_hypotheses

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
