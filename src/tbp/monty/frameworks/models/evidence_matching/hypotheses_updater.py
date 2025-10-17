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
from typing import Any, Dict, Literal, Optional, Protocol

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
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    ConsistentHypothesesIds,
)
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_initial_possible_poses,
    possible_sensed_directions,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
)

logger = logging.getLogger(__name__)

HypothesesUpdateTelemetry = Optional[Dict[str, Any]]
HypothesesUpdaterTelemetry = Dict[str, Any]


class HypothesesUpdater(Protocol):
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

        Args:
            hypotheses: Hypotheses for all input channels for the graph_id
            features: Input features
            displacements: Given displacements
            graph_id: Identifier of the graph being updated
            mapper: Mapper for the graph_id to extract data from
                evidence, locations, and poses based on the input channel
            evidence_update_threshold: Evidence update threshold

        Returns:
            The list of channel hypotheses updates to be applied.
        """
        ...

    def remap_hypotheses_ids_to_present(
        self, hypotheses_ids: ConsistentHypothesesIds
    ) -> ConsistentHypothesesIds:
        """Update hypotheses ids based on resizing of hypothesis space.

        Args:
            hypotheses_ids: Hypotheses ids to be updated

        Returns:
            The list of the updated hypotheses ids.
        """
        ...


class DefaultHypothesesUpdater:
    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        tolerances: dict,
        evidence_threshold_config: float | str = "all",
        feature_evidence_calculator: type[FeatureEvidenceCalculator] = (
            DefaultFeatureEvidenceCalculator
        ),
        feature_evidence_increment: int = 1,
        features_for_matching_selector: type[FeaturesForMatchingSelector] = (
            DefaultFeaturesForMatchingSelector
        ),
        initial_possible_poses: Literal["uniform", "informed"]
        | list[Rotation] = "informed",
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
        umbilical_num_poses: int = 8,
    ):
        """Initializes the DefaultHypothesesUpdater.

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
                should be updated. When this parameter is either '[int]%' or
                'x_percent_threshold', then this parameter is applied to the evidence
                for the Most Likely Hypothesis (MLH) to determine a minimum evidence
                threshold in order for other hypotheses to be updated. Any hypotheses
                falling below the resulting evidence threshold do not get updated. The
                other options set a fixed threshold that does not take MLH evidence into
                account. In [int, float, '[int]%', 'mean', 'median', 'all',
                'x_percent_threshold']. Defaults to 'all'.
            feature_evidence_calculator: Class to
                calculate feature evidence for all nodes. Defaults to the default
                calculator.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence as opposed to the present_weight). Defaults to 1.
            features_for_matching_selector: Class to
                select if features should be used for matching. Defaults to the default
                selector.
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
            The list of hypotheses updates to be applied to each input channel.
        """
        # Get all usable input channels
        # NOTE: We might also want to check the confidence in the input channel
        # features. This information is currently not available here.
        # TODO S: Once we pull the observation class into the LM we could add this.
        input_channels_to_use = all_usable_input_channels(
            features, self.graph_memory.get_input_channels_in_graph(graph_id)
        )

        if len(input_channels_to_use) == 0:
            logger.info(
                f"No input channels observed for {graph_id} that are stored in the "
                "model. Not updating evidence."
            )
            return []

        hypotheses_updates = []

        for input_channel in input_channels_to_use:
            # Determine if the hypothesis space exists
            initialize_hyp_space = bool(input_channel not in mapper.channels)

            # Initialize a new hypothesis space using graph nodes
            if initialize_hyp_space:
                # TODO H: When initializing a hypothesis for a channel later on (if
                # displacement is not None), include most likely existing hypothesis
                # from other channels?
                channel_possible_hypotheses = self._get_initial_hypothesis_space(
                    channel_features=features[input_channel],
                    graph_id=graph_id,
                    input_channel=input_channel,
                )
            # Retrieve existing hypothesis space for a specific input channel
            else:
                channel_hypotheses = mapper.extract_hypotheses(
                    hypotheses, input_channel
                )

                # We only displace existing hypotheses since the newly sampled
                # hypotheses should not be affected by the displacement from the last
                # sensory input.
                channel_possible_hypotheses = (
                    self.hypotheses_displacer.displace_hypotheses_and_compute_evidence(
                        channel_displacement=displacements[input_channel],
                        channel_features=features[input_channel],
                        evidence_update_threshold=evidence_update_threshold,
                        graph_id=graph_id,
                        possible_hypotheses=channel_hypotheses,
                        total_hypotheses_count=hypotheses.evidence.shape[0],
                    )
                )

            hypotheses_updates.append(channel_possible_hypotheses)

        return hypotheses_updates, None

    def _get_all_informed_possible_poses(
        self, graph_id: str, sensed_channel_features: dict, input_channel: str
    ):
        """Initialize hypotheses on possible rotations for each location.

        Similar to _get_informed_possible_poses but doesn't require looping over nodes

        For this we use the surface normal and curvature directions and check how
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

        logger.debug(f"Determining possible poses using input from {input_channel}")
        node_directions = self.graph_memory.get_rotation_features_at_all_nodes(
            graph_id, input_channel
        )
        sensed_directions = sensed_channel_features["pose_vectors"]
        # Check if PCs in patch are similar -> need to sample more directions
        if (
            "pose_fully_defined" in sensed_channel_features.keys()
            and not sensed_channel_features["pose_fully_defined"]
        ):
            possible_s_d = possible_sensed_directions(
                sensed_directions, self.umbilical_num_poses
            )
        else:
            possible_s_d = possible_sensed_directions(sensed_directions, 2)

        for s_d in possible_s_d:
            # Since we have orthonormal vectors and know their correspondence we can
            # directly calculate the rotation instead of using the Kabsch estimate
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

    def _get_initial_hypothesis_space(
        self, channel_features: dict, graph_id: str, input_channel: str
    ) -> ChannelHypotheses:
        if self.initial_possible_poses is None:
            # Get initial poses for all locations informed by pose features
            (
                initial_possible_channel_locations,
                initial_possible_channel_rotations,
            ) = self._get_all_informed_possible_poses(
                graph_id, channel_features, input_channel
            )
        else:
            initial_possible_channel_locations = []
            initial_possible_channel_rotations = []
            all_channel_locations = self.graph_memory.get_locations_in_graph(
                graph_id, input_channel
            )
            # Initialize fixed possible poses (without using pose features)
            for rotation in self.initial_possible_poses:
                for node_id in range(len(all_channel_locations)):
                    initial_possible_channel_locations.append(
                        all_channel_locations[node_id]
                    )
                    initial_possible_channel_rotations.append(rotation.as_matrix())

            initial_possible_channel_locations = np.array(
                initial_possible_channel_locations
            )
            initial_possible_channel_rotations = np.array(
                initial_possible_channel_rotations
            )
        # There will always be two feature weights (surface normal and curvature
        # direction). If there are no more weight we are not using features for
        # matching and skip this step. Doing matching with only morphology can
        # currently be achieved in two ways. Either we don't specify tolerances
        # and feature_weights or we set the global feature_evidence_increment to 0.
        if self.use_features_for_matching[input_channel]:
            # Get real valued features match for each node
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
        return ChannelHypotheses(
            input_channel=input_channel,
            evidence=evidence,
            locations=initial_possible_channel_locations,
            poses=initial_possible_channel_rotations,
        )

    def remap_hypotheses_ids_to_present(
        self, hypotheses_ids: ConsistentHypothesesIds
    ) -> ConsistentHypothesesIds:
        """Update hypotheses ids based on resizing of hypothesis space.

        We do not resize the hypotheses space when using `DefaultHypothesesUpdater`,
        therefore, we return the same ids without update.

        Args:
            hypotheses_ids: Hypotheses ids to be updated

        Returns:
            The list of the updated hypotheses ids.
        """
        return hypotheses_ids


def all_usable_input_channels(
    features: dict, all_input_channels: list[str]
) -> list[str]:
    """Determine all usable input channels.

    Args:
        features: Input features.
        all_input_channels: All input channels that are stored in the graph.

    Returns:
        All input channels that are usable for matching.
    """
    # Get all usable input channels
    # NOTE: We might also want to check the confidence in the input channel
    # features. This information is currently not available here.
    # TODO S: Once we pull the observation class into the LM we could add this.
    return [ic for ic in features.keys() if ic in all_input_channels]
