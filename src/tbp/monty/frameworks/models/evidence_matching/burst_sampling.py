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

from tbp.monty.frameworks.models.evidence_matching.channels import (
    all_usable_input_channels,
)
from tbp.monty.frameworks.models.evidence_matching.evidence_slope_tracker import (
    EvidenceSlopeTracker,
    HypothesesSelection,
)
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
from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.models.evidence_matching.hypotheses_displacer import (
    DefaultHypothesesDisplacer,
    HypothesisDisplacerTelemetry,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    HypothesesUpdateTelemetry,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
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
class BurstSamplingTelemetry:
    """Telemetry for burst sampling hypotheses updater.

    Stores which hypotheses were removed or added at the current step.

    Note:
        TODO: Additional description here regarding availability of hypotheses
        identified by `removed_ids`.
    """

    displacer_telemetry: HypothesisDisplacerTelemetry
    added_ids: npt.NDArray[np.int_]
    ages: npt.NDArray[np.int_]
    evidence_slopes: npt.NDArray[np.float64]
    removed_ids: npt.NDArray[np.int_]
    max_slope: float


class BurstSamplingHypothesesUpdater:
    """Hypotheses updater that adds and deletes hypotheses based on evidence slope.

    This updater enables updating of the hypothesis space by intelligently sampling
    and rebuilding the hypothesis space when the model's prediction error is high. The
    prediction error is determined based on the highest evidence slope over all the
    objects hypothesis spaces. If the hypothesis with the highest slope is unable to
    accumulate evidence at a high enough slope, i.e., none of the current hypotheses
    match the incoming observations well, a sampling burst is triggered. A sampling
    burst adds new hypotheses over a specified `sampling_burst_duration` number of
    consecutive steps to all hypothesis spaces. This burst duration reduces the effect
    of sensor noise. Hypotheses are deleted when their smoothed evidence slope is below
    `deletion_trigger_slope`.

    The burst sampling process is governed by four main parameters:
      - `sampling_multiplier`: Determines the number of hypotheses to sample during
        bursts as a multiplier of the object graph nodes.
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
        """Initializes the BurstSamplingHypothesesUpdater.

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
                should be updated. In the `BurstSamplingHypothesesUpdater` we always
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
            sampling_multiplier: Determines the number of hypotheses to sample during
                bursts as a multiplier of the object graph nodes. Value of 0.0 results
                in no sampling. Value can be greater than 1 but not to exceed the
                `num_hyps_per_node` of the current step. Defaults to 0.4.
            deletion_trigger_slope: Hypotheses below this threshold are deleted.
                Expected range matches the range of step evidence change, i.e.,
                [-1.0, 2.0]. Defaults to 0.5.
            sampling_burst_duration: The number of steps in every sampling burst.
                Defaults to 5.
            burst_trigger_slope: A threshold below which a sampling burst is triggered.
                Defaults to 1.0.
            include_telemetry: Flag to control if we want to calculate and return the
                burst sampling telemetry in the `update_hypotheses` method. Defaults to
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
                "'all' for `BurstSamplingHypothesesUpdater`"
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

        if self.sampling_multiplier < 0:
            raise ValueError("sampling_multiplier should be >= 0")

        self.reset()

    def reset(self) -> None:
        self.sampling_burst_steps = 0

        # Dictionary of slope trackers, one for each graph_id
        self.evidence_slope_trackers: dict[str, EvidenceSlopeTracker] = {}

    def __enter__(self) -> Self:
        """Enter context manager, runs before updating the hypotheses.

        We calculate the max slope and update burst sampling parameters before running
        the hypotheses update loop/threads over all the graph_ids and channels.

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
        displacement: npt.NDArray[np.float64] | None,
        graph_id: str,
        evidence_update_threshold: float,
    ) -> tuple[Hypotheses | None, HypothesesUpdateTelemetry]:
        """Update hypotheses based on sensor displacement and sensed features.

        Updates the existing hypothesis space or initializes a new hypothesis space
        if one does not exist (i.e., at the beginning of the episode). Updating the
        hypothesis space includes displacing the hypotheses' possible locations, as well
        as updating their evidence scores. Evidence from all available input channels
        is computed and summed by the displacer.

        Args:
            hypotheses: Hypothesis space for the graph id.
            features: Input features keyed by channel name.
            displacement: LM displacement between the current and previous input.
            graph_id: Identifier of the graph being updated.
            evidence_update_threshold: Evidence update threshold.

        Returns:
            A tuple containing the updated hypotheses (or None if no channels available)
            and hypotheses update telemetry for analysis. The hypotheses update
            telemetry is a dictionary containing:
                - added_ids: IDs of hypotheses added during burst sampling at the
                    current timestep.
                - ages: The ages of hypotheses as tracked by the `EvidenceSlopeTracker`.
                - evidence_slopes: The slopes extracted from the `EvidenceSlopeTracker`.
                - removed_ids: IDs of hypotheses removed. Note that these IDs can only
                    be used to index hypotheses from the previous timestep.
        """
        # Initialize a `EvidenceSlopeTracker` to keep track of evidence slopes
        # for hypotheses of a specific graph_id
        if graph_id not in self.evidence_slope_trackers:
            self.evidence_slope_trackers[graph_id] = EvidenceSlopeTracker()
        tracker = self.evidence_slope_trackers[graph_id]

        input_channels_to_use = all_usable_input_channels(
            features, self.graph_memory.get_input_channels_in_graph(graph_id)
        )

        if len(input_channels_to_use) == 0:
            return None, {}

        hypotheses_selection, informed_samples_per_channel = self._sample_count(
            features=features,
            graph_id=graph_id,
            input_channels=input_channels_to_use,
            tracker=tracker,
        )

        existing_hypotheses = self._sample_existing(
            hypotheses_selection=hypotheses_selection,
            hypotheses=hypotheses,
            tracker=tracker,
        )
        informed_hypotheses = self._sample_informed(
            features=features,
            graph_id=graph_id,
            samples_per_channel=informed_samples_per_channel,
            tracker=tracker,
        )

        # We only displace existing hypotheses since the newly sampled hypotheses
        # should not be affected by the displacement from the last sensory input.
        if len(hypotheses_selection.ids_to_retain):
            existing_hypotheses, displacer_telemetry = (
                self.hypotheses_displacer.displace_hypotheses_and_compute_evidence(
                    displacement=displacement,
                    features=features,
                    evidence_update_threshold=evidence_update_threshold,
                    graph_id=graph_id,
                    possible_hypotheses=existing_hypotheses,
                )
            )
        else:
            displacer_telemetry = HypothesisDisplacerTelemetry(
                mlh_prediction_error=None
            )

        hypotheses_update = Hypotheses.concatenate(
            [existing_hypotheses, informed_hypotheses]
        )
        tracker.update(hypotheses_update.evidence)

        if self.include_telemetry:
            telemetry = asdict(
                BurstSamplingTelemetry(
                    displacer_telemetry=displacer_telemetry,
                    added_ids=(
                        np.arange(len(hypotheses_update.evidence))[
                            -len(informed_hypotheses.evidence) :
                        ]
                        if len(informed_hypotheses.evidence) > 0
                        else np.array([], dtype=np.int_)
                    ),
                    ages=tracker.hyp_ages(),
                    evidence_slopes=tracker.calculate_slopes(),
                    removed_ids=hypotheses_selection.ids_to_remove,
                    max_slope=self.max_slope,
                )
            )
        else:
            telemetry = {
                "mlh_prediction_error": displacer_telemetry.mlh_prediction_error,
            }

        return hypotheses_update, telemetry

    def _num_hyps_per_node(self, features: dict[str, Any]) -> int:
        """Calculate the number of hypotheses per node.

        Args:
            features: Features for the input channel.

        Returns:
            The number of hypotheses per node.
        """
        if self.initial_possible_poses is None:
            return 2 if features["pose_fully_defined"] else self.umbilical_num_poses

        return len(self.initial_possible_poses)

    def _sample_count(
        self,
        features: dict,
        graph_id: str,
        input_channels: list[str],
        tracker: EvidenceSlopeTracker,
    ) -> tuple[HypothesesSelection, dict[str, int]]:
        """Calculates the number of existing and informed hypotheses needed.

        Args:
            features: Input features keyed by channel name.
            graph_id: Identifier of the graph being queried.
            input_channels: Usable input channels for this graph.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            A tuple containing the hypotheses selection and a dictionary mapping
            each channel to the number of informed hypotheses to sample from it.
            Hypotheses selection are retained from existing ones while new
            hypotheses will be initialized, informed by pose sensory information.

        Notes:
            This function takes into account the following parameters:
              - `sampling_multiplier`: The number of hypotheses to sample during bursts.
                This is defined as a multiplier of the number of nodes in the object
                graph.
              - `deletion_trigger_slope`: This dictates how many hypotheses to
                delete. Hypotheses below this threshold are deleted.
              - `sampling_burst_steps`: The remaining number of burst steps. This value
                is decremented in the `post_step` function.
        """
        informed_samples_per_channel: dict[str, int] = {}
        if self.sampling_burst_steps > 0:
            # Calculate informed count per channel based on each channel's
            # num_hyps_per_node and node count. Each channel may have a
            # different num_hyps_per_node (e.g. due to pose_fully_defined),
            # so we compute per-channel totals to ensure each is divisible
            # by its own num_hyps_per_node.
            for channel in input_channels:
                num_nodes = self.graph_memory.get_locations_in_graph(
                    graph_id, channel
                ).shape[0]
                num_hyps_per_node = self._num_hyps_per_node(features[channel])

                # This makes sure that we do not request more than the available
                # number of informed hypotheses
                capped_multiplier = min(self.sampling_multiplier, num_hyps_per_node)

                # Calculate the total number of informed hypotheses to be sampled
                sample_count = round(num_nodes * capped_multiplier)

                # Ensure divisible by this channel's num_hyps_per_node
                sample_count -= sample_count % num_hyps_per_node

                informed_samples_per_channel[channel] = sample_count

        # Returns a selection of hypotheses to retain/delete
        hypotheses_selection = (
            tracker.select_hypotheses(
                slope_threshold=self.deletion_trigger_slope,
            )
            if tracker.total_size() > 0
            else HypothesesSelection(mask_to_retain=[])
        )

        return (
            hypotheses_selection,
            informed_samples_per_channel,
        )

    def _sample_existing(
        self,
        hypotheses_selection: HypothesesSelection,
        hypotheses: Hypotheses,
        tracker: EvidenceSlopeTracker,
    ) -> Hypotheses:
        """Samples the specified number of existing hypotheses to retain.

        Note that we are not sampling the existing hypotheses in a probabilistic
        sense (e.g., random or seed-generation). Instead, those are deterministically
        determined using the slope tracker and the deletion threshold, then retained
        by filtering the list of existing hypotheses.

        Args:
            hypotheses_selection: The selection of hypotheses to retain/remove.
            hypotheses: Hypothesis space for the graph_id.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            The sampled existing hypotheses.
        """
        ids_to_retain = hypotheses_selection.ids_to_retain

        # Return empty arrays for no hypotheses to sample
        if len(ids_to_retain) == 0:
            # Clear all hypotheses from the tracker
            tracker.clear_hyp()

            return Hypotheses(
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
                possible=np.zeros(0, dtype=np.bool_),
            )

        # Update tracker by removing the ids_to_remove
        tracker.remove_hyp(hypotheses_selection.ids_to_remove)

        return Hypotheses(
            locations=hypotheses.locations[ids_to_retain],
            poses=hypotheses.poses[ids_to_retain],
            evidence=hypotheses.evidence[ids_to_retain],
            possible=hypotheses.possible[ids_to_retain],
        )

    def _sample_informed(
        self,
        features: dict,
        graph_id: str,
        samples_per_channel: dict[str, int],
        tracker: EvidenceSlopeTracker,
    ) -> Hypotheses:
        """Samples the specified number of fully informed hypotheses.

        For each channel, the method identifies the top-k nodes with the highest
        feature match scores to the current step's input and samples hypotheses only
        from those nodes. The number of hypotheses per channel is pre-computed
        by ``_sample_count``.

        The sampling includes:
          - Selecting the top-k node indices based on evidence scores, where k is
             determined by the channel's informed count and its num_hyps_per_node.
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
            features: Input features keyed by channel name.
            graph_id: Identifier of the graph being queried.
            samples_per_channel: Dictionary mapping each channel to the number
                of informed hypotheses to sample from it, as computed by
                ``_sample_count``.
            tracker: Slope tracker for the evidence values of a
                graph_id

        Returns:
            The sampled informed hypotheses.

        """
        sampled_hypotheses = []
        for channel, sample_count in samples_per_channel.items():
            if sample_count == 0:
                continue
            sampled_hypotheses.append(
                self._sample_from_channel(
                    features=features[channel],
                    count=sample_count,
                    graph_id=graph_id,
                    input_channel=channel,
                )
            )

        # Return empty arrays for no hypotheses to sample
        if not sampled_hypotheses:
            return Hypotheses(
                locations=np.zeros((0, 3)),
                poses=np.zeros((0, 3, 3)),
                evidence=np.zeros(0),
                possible=np.zeros(0, dtype=np.bool_),
            )

        hypotheses = Hypotheses.concatenate(sampled_hypotheses)

        # Add hypotheses to slope trackers
        tracker.add_hyp(hypotheses.count)

        return hypotheses

    def _sample_from_channel(
        self,
        features: dict[str, Any],
        count: int,
        graph_id: str,
        input_channel: str,
    ) -> Hypotheses:
        """Sample informed hypotheses from a single channel's graph.

        Args:
            features: Features for this input channel.
            count: Number of hypotheses to sample from this channel.
            graph_id: Identifier of the graph being queried.
            input_channel: The channel to sample from.

        Returns:
            Hypotheses sampled from this channel.
        """
        num_hyps_per_node = self._num_hyps_per_node(features)
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
                channel_query_features=features,
                channel_tolerances=self.tolerances[input_channel],
                input_channel=input_channel,
            )
            # Find the indices for the nodes with highest evidence scores. The sorting
            # is done in ascending order, so extract the indices from the end of
            # the argsort array. We get the needed number of informed nodes not
            # the number of needed hypotheses.
            top_indices = np.argsort(node_feature_evidence)[
                -int(count // num_hyps_per_node) :
            ]
            node_feature_evidence_filtered = (
                node_feature_evidence[top_indices] * self.feature_evidence_increment
            )
        else:
            num_nodes = self.graph_memory.get_locations_in_graph(
                graph_id, input_channel
            ).shape[0]
            top_indices = np.arange(num_nodes)[: int(count // num_hyps_per_node)]
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
            sensed_directions = features["pose_vectors"]
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
        possible = np.zeros_like(selected_feature_evidence, dtype=np.bool_)

        return Hypotheses(
            locations=selected_locations,
            poses=selected_rotations,
            evidence=selected_feature_evidence,
            possible=possible,
        )

    def _max_global_slope(self) -> float:
        """Compute the maximum slope over all objects.

        Returns:
            The maximum global slope if finite, otherwise -np.inf
        """
        max_slope = -np.inf

        for tracker in self.evidence_slope_trackers.values():
            if tracker.total_size() == 0:
                continue

            slopes = tracker.calculate_slopes()
            finite_slopes = slopes[np.isfinite(slopes)]
            if finite_slopes.size:
                max_slope = max(max_slope, np.max(finite_slopes))

        return max_slope
