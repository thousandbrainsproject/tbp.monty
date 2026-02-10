# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from typing_extensions import Self

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
    HypothesisDisplacerTelemetry,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    ChannelHypothesesUpdateTelemetry,
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
class ChannelHypothesesResamplingTelemetry(ChannelHypothesesUpdateTelemetry):
    """Hypotheses resampling telemetry for a channel.

    For a given input channel, this class stores which hypotheses were removed or
    added at the current step.

    Note:
        TODO: Additional description here regarding availability of hypotheses
        identified by `removed_ids`.
    """

    channel_hypothesis_displacer_telemetry: HypothesisDisplacerTelemetry
    added_ids: npt.NDArray[np.int_]
    ages: npt.NDArray[np.int_]
    evidence_slopes: npt.NDArray[np.float64]
    removed_ids: npt.NDArray[np.int_]
    max_slope: float


class ResamplingHypothesesUpdater:
    """Hypotheses updater that adds and deletes hypotheses based on evidence slope.

    This updater enables updating of the hypothesis space by intelligently resampling
    and rebuilding the hypothesis space when the model's prediction error is high. The
    prediction error is determined based on the highest evidence slope over all the
    objects hypothesis spaces. If the hypothesis with the highest slope is unable to
    accumulate evidence at a high enough slope, i.e., none of the current hypotheses
    match the incoming observations well, a sampling burst is triggered. A sampling
    burst adds new hypotheses over a specified `sampling_burst_duration` number of
    consecutive steps to all hypothesis spaces. This burst duration reduces the effect
    of sensor noise. Hypotheses are deleted when their smoothed evidence slope is below
    `deletion_trigger_slope`.

    The resampling process is governed by four main parameters:
      - `sampling_multiplier`: Determines the number of hypotheses to resample
        as a multiplier of the object graph nodes.
      - `deletion_trigger_slope`: Hypotheses below this threshold are deleted.
      - `sampling_burst_duration`: The number of consecutive steps in each burst.
      - `burst_trigger_slope`: The threshold for triggering a sampling burst. This
        threshold is applied to the highest global slope over all the hypotheses (i.e.,
        over all objects' hypothesis spaces). The range of this slope is [-1, 2].

    To reproduce the behavior of `DefaultHypothesesUpdater` sampling a fixed number of
    hypotheses only at the beginning of the episode, you can set:
        - `sampling_multiplier=2` (or `umbilical_num_poses` if PC undefined)
        - `deletion_trigger_slope=-np.inf` (no deletion is allowed)
        - `sampling_burst_duration=1` (sample the full burst over a single step)
        - `burst_trigger_slope=-np.inf` (never trigger additional bursts)

    These parameters will trigger a single-step burst at the first step of the episode.
    Note that if the PC of the first observation is undetermined,
    `sampling_multiplier` should be set to the value of `umbilical_num_poses` to
    reproduce the exact results of `DefaultHypothesesUpdater`. In practice, this is
    difficult to predict because it relies on the first sampled observation.
    """

    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        evidence_threshold_config: Literal["all"],
        feature_evidence_calculator: type[FeatureEvidenceCalculator] = (
            DefaultFeatureEvidenceCalculator
        ),
        feature_evidence_increment: int = 1,
        features_for_matching_selector: type[FeaturesForMatchingSelector] = (
            DefaultFeaturesForMatchingSelector
        ),
        sampling_multiplier: float = 0.4,
        deletion_trigger_slope: float = 0.5,
        sampling_burst_duration: int = 5,
        burst_trigger_slope: float = 1.0,
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
            feature_weights: How much each feature should be weighted when
                calculating the evidence update for a hypothesis. Weights are stored
                in a dictionary with keys corresponding to features (same as keys in
                tolerances).
            graph_memory: The graph memory to read graphs from.
            max_match_distance: Maximum distance between a tested location and a stored
                location for them to be matched.
            tolerances: How much each observed feature can deviate from the
                stored features while still being considered a match.
            evidence_threshold_config: How to decide which hypotheses
                should be updated. In the `ResamplingHypothesesUpdater` we always
                update 'all' hypotheses. Hypotheses with decreasing evidence are deleted
                instead of excluded from updating. Must be set to 'all'.
            feature_evidence_calculator: Class to calculate feature evidence for all
                nodes. Defaults to the default calculator.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence, unlike the present_weight). Defaults to 1.
            features_for_matching_selector: Class to
                select if features should be used for matching. Defaults to the default
                selector.
            sampling_multiplier: Determines the number of hypotheses to resample
                as a multiplier of the object graph nodes. Value of 0.0 results in no
                resampling. Value can be greater than 1 but not to exceed the
                `num_hyps_per_node` of the current step. Defaults to 0.4.
            deletion_trigger_slope: Hypotheses below this threshold are deleted.
                Expected range matches the range of step evidence change, i.e.,
                [-1.0, 2.0]. Defaults to 0.5.
            sampling_burst_duration: The number of steps in every sampling burst.
                Defaults to 5.
            burst_trigger_slope: A threshold below which a sampling burst is triggered.
                Defaults to 1.0.
            include_telemetry: Flag to control if we want to calculate and return the
                resampling telemetry in the `update_hypotheses` method. Defaults to
                False.
            initial_possible_poses: Initial
                possible poses to test. Defaults to "informed".
            max_nneighbors: Maximum number of nearest neighbors to consider in the
                radius of a hypothesis for calculating the evidence. Defaults to 3.
            past_weight: How much the evidence accumulated so far should be
                weighted when combined with the evidence from the most recent
                observation. Defaults to 1.
            present_weight: How much the current evidence should be weighted
                when added to the previous evidence. If past_weight and present_weight
                add up to 1, the evidence is bounded and can't grow infinitely. Defaults
                to 1.
                NOTE: Right now this doesn't give as good performance as with unbounded
                evidence since we don't keep a full history of what we saw. With a more
                efficient policy and better parameters, that may be possible to use and
                could help when moving from one object to another and generally make
                setting thresholds more intuitive.
            umbilical_num_poses: Number of sampled rotations in the direction of
                the plane perpendicular to the surface normal. These are sampled at
                umbilical points (i.e., points where PC directions are undefined).

        Raises:
            ValueError: If the sampling_multiplier is less than 0
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
        self.sampling_multiplier = sampling_multiplier
        self.deletion_trigger_slope = deletion_trigger_slope
        self.sampling_burst_duration = sampling_burst_duration
        self.burst_trigger_slope = burst_trigger_slope
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
        if self.sampling_multiplier < 0:
            raise ValueError("sampling_multiplier should be >= 0")

        self.reset()

    def reset(self) -> None:
        self.sampling_burst_steps = 0

        # Dictionary of slope trackers, one for each graph_id
        self.evidence_slope_trackers: dict[str, EvidenceSlopeTracker] = {}

    def __enter__(self) -> Self:
        """Enter context manager, runs before updating the hypotheses.

        We calculate the max slope and update resampling parameters before running the
        hypotheses update loop/threads over all the graph_ids and channels.

        Returns:
            Self: The context manager instance.
        """
        self.max_slope = self._max_global_slope()

        if (
            self.max_slope <= self.burst_trigger_slope
            and self.sampling_burst_steps == 0
        ):
            self.sampling_burst_steps = self.sampling_burst_duration

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, runs after updating the hypotheses.

        We decrement the burst steps by 1 every step for the duration of the burst.
        """
        if not exc_type and self.sampling_burst_steps > 0:
            self.sampling_burst_steps -= 1

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

        Updates the existing hypothesis space or initializes a new hypothesis space
        if one does not exist (i.e., at the beginning of the episode). Updating the
        hypothesis space includes displacing the hypotheses' possible locations, as well
        as updating their evidence scores. This process is repeated for each input
        channel in the graph.

        Args:
            hypotheses: Hypotheses for all input channels in the graph
            features: Input features
            displacements: Given displacements
            graph_id: Identifier of the graph being updated
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel
            evidence_update_threshold: Evidence update threshold.

        Returns:
            A tuple containing the list of hypothesis updates to be applied to each
            input channel and hypotheses update telemetry for analysis. The hypotheses
            update telemetry is a dictionary containing:
                - added_ids: IDs of hypotheses added during resampling at the current
                    timestep.
                - ages: The ages of hypotheses as tracked by the `EvidenceSlopeTracker`.
                - evidence_slopes: The slopes extracted from the `EvidenceSlopeTracker`.
                - removed_ids: IDs of hypotheses removed during resampling. Note that
                    these IDs can only be used to index hypotheses from the previous
                    time step.
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
        channel_hypothesis_displacer_telemetry: dict[
            str, HypothesisDisplacerTelemetry
        ] = {}

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
            if len(hypotheses_selection.maintain_ids):
                existing_hypotheses, channel_hypothesis_displacer_telemetry = (
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
                possible=np.hstack(
                    [existing_hypotheses.possible, informed_hypotheses.possible]
                ),
            )
            hypotheses_updates.append(channel_hypotheses)

            # Update tracker evidence
            tracker.update(channel_hypotheses.evidence, input_channel)

            # Telemetry update
            resampling_telemetry[input_channel] = asdict(
                ChannelHypothesesResamplingTelemetry(
                    channel_hypothesis_displacer_telemetry=channel_hypothesis_displacer_telemetry,
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
                    max_slope=self.max_slope,
                )
            )

        # Still return prediction error.
        # TODO: make this nicer like dependent on log_level.
        if not self.include_telemetry:
            updater_telemetry = {
                k: asdict(
                    ChannelHypothesesUpdateTelemetry(
                        channel_hypothesis_displacer_telemetry=v[
                            "channel_hypothesis_displacer_telemetry"
                        ]
                    )
                )
                for k, v in resampling_telemetry.items()
            }

        return (
            hypotheses_updates,
            resampling_telemetry if self.include_telemetry else updater_telemetry,
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

        return len(self.initial_possible_poses)

    def _sample_count(
        self,
        input_channel: str,
        channel_features: dict,
        graph_id: str,
        mapper: ChannelMapper,
        tracker: EvidenceSlopeTracker,
    ) -> tuple[HypothesesSelection, int]:
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
              - `sampling_multiplier`: The number of hypotheses to resample. This
                is defined as a multiplier of the number of nodes in the object graph.
              - `deletion_trigger_slope`: This dictates how many hypotheses to
                delete. Hypotheses below this threshold are deleted.
              - `sampling_burst_steps`: The remaining number of burst steps. This value
                is decremented in the `post_step` function.
        """
        new_informed = 0
        if self.sampling_burst_steps > 0:
            graph_num_points = self.graph_memory.get_locations_in_graph(
                graph_id, input_channel
            ).shape[0]
            num_hyps_per_node = self._num_hyps_per_node(channel_features)

            # This makes sure that we do not request more than the available number of
            # informed hypotheses
            sampling_multiplier = min(self.sampling_multiplier, num_hyps_per_node)

            # Calculate the total number of informed hypotheses to be resampled
            new_informed = round(graph_num_points * sampling_multiplier)

            # Ensure the `new_informed` is divisible by `num_hyps_per_node`
            new_informed -= new_informed % num_hyps_per_node

        # Returns a selection of hypotheses to maintain/delete
        hypotheses_selection = (
            tracker.select_hypotheses(
                slope_threshold=self.deletion_trigger_slope, channel=input_channel
            )
            if input_channel in mapper.channels
            else HypothesesSelection(maintain_mask=[])
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
    ) -> ChannelHypotheses:
        """Samples the specified number of existing hypotheses to retain.

        Note that we are not sampling the existing hypotheses in a probabilistic
        sense (e.g., random or seed-generation). Instead, those are deterministically
        determined using the slope tracker and the deletion threshold, then maintained
        by filtering the list of existing hypotheses.

        Args:
            hypotheses_selection: The selection of hypotheses to maintain/remove.
            hypotheses: Hypotheses for all input channels in the graph_id.
            input_channel: The channel for which to sample existing hypotheses.
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            The sampled existing hypotheses.
        """
        maintain_ids = hypotheses_selection.maintain_ids

        # Return empty arrays for no hypotheses to sample
        if len(maintain_ids) == 0:
            # Clear all channel hypotheses from the tracker
            tracker.clear_hyp(input_channel)

            return ChannelHypotheses(
                input_channel=input_channel,
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
                possible=np.zeros(0, dtype=np.bool_),
            )

        # Update tracker by removing the remove_ids
        tracker.remove_hyp(hypotheses_selection.remove_ids, input_channel)

        channel_hypotheses = mapper.extract_hypotheses(hypotheses, input_channel)
        return ChannelHypotheses(
            input_channel=channel_hypotheses.input_channel,
            locations=channel_hypotheses.locations[maintain_ids],
            poses=channel_hypotheses.poses[maintain_ids],
            evidence=channel_hypotheses.evidence[maintain_ids],
            possible=channel_hypotheses.possible[maintain_ids],
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
                possible=np.zeros(0, dtype=np.bool_),
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

        # Newly sampled hypotheses cannot be marked as possible
        possible_hyp = np.zeros_like(selected_feature_evidence, dtype=np.bool_)

        # Add hypotheses to slope trackers
        tracker.add_hyp(selected_feature_evidence.shape[0], input_channel)

        return ChannelHypotheses(
            input_channel=input_channel,
            locations=selected_locations,
            poses=selected_rotations,
            evidence=selected_feature_evidence,
            possible=possible_hyp,
        )

    def _max_global_slope(self) -> float:
        """Compute the maximum slope over all objects and channels.

        Returns:
            The maximum global slope if finite, otherwise float("nan")
        """
        max_slope = float("-inf")

        for tracker in self.evidence_slope_trackers.values():
            for channel in tracker.evidence_buffer:
                if tracker.total_size(channel) == 0:
                    continue

                slopes = tracker.calculate_slopes(channel)
                finite_slopes = slopes[np.isfinite(slopes)]
                if finite_slopes.size:
                    max_slope = max(max_slope, np.max(finite_slopes))

        return max_slope
