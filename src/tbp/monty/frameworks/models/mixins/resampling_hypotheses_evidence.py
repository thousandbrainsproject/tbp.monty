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
    possible_sensed_directions,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_multiple_orthonormal_vectors,
)


class ResamplingHypothesesEvidenceMixin:
    """Mixin that adds resampling capability to EvidenceGraph learning modules.

    This mixin enables updating of the hypothesis space by resampling and rebuilding
    the hypothesis space at every step. We resample hypotheses from the existing
    hypothesis space, as well as new hypotheses informed by the sensed pose.

    The resampling process is governed by two main parameters:
      - `hypotheses_count_multiplier`: scales the total number of hypotheses every step.
      - `hypotheses_existing_to_new_ratio`: controls the proportion of existing vs.
          informed hypotheses during resampling.

    To reproduce the original behavior of `EvidenceGraphLM` sampling a fixed number of
    hypotheses only at the beginning of the episode, you can set
    `hypotheses_count_multiplier=1.0` and `hypotheses_existing_to_new_ratio=0.0`.

    Compatible with:
        - EvidenceGraphLM

    Raises:
        TypeError: If used in a class that is not a subclass of `EvidenceGraphLM`.
    """

    def __init__(
        self,
        *args: object,
        hypotheses_count_multiplier=1.0,
        hypotheses_existing_to_new_ratio=0.0,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)

        # Controls the shrinking or growth of hypothesis space size
        # Cannot be less than 0
        self.hypotheses_count_multiplier = max(0, hypotheses_count_multiplier)

        # Controls the ratio of existing to newly sampled hypotheses
        # Bounded between 0 and 1
        self.hypotheses_existing_to_new_ratio = max(
            0, min(hypotheses_existing_to_new_ratio, 1)
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

    def _num_hyps_per_node(self, features, input_channel):
        if self.initial_possible_poses is None:
            return (
                2
                if features[input_channel]["pose_fully_defined"]
                else self.umbilical_num_poses
            )
        else:
            return len(self.initial_possible_poses)

    def _update_evidence(
        self,
        features: Dict,
        displacements: Optional[Dict],
        graph_id: str,
    ) -> None:
        """Update evidence of hypotheses space with resampling.

        Updates existing hypotheses space by:
            1. Calculating sample count for existing and informed hypotheses
            2. Sampling hypotheses for existing and informed hypotheses types
            3. Displacing (and updating evidence of) existing hypotheses using
                given displacements and sensed features
            4. Concatenating all samples (existing + new) to rebuild the hypothesis
                space

        This process is repeated for each input channel in the graph.

        Args:
            features (dict): input features
            displacements (dict or None): given displacements
            graph_id (str): identifier of the graph being updated
        """
        start_time = time.time()

        # Initialize a `ChannelMapper` to keep track of input channel range
        # of hypotheses for a specific graph_id
        if graph_id not in self.channel_hypothesis_mapping:
            self.channel_hypothesis_mapping[graph_id] = ChannelMapper()

        # Get all usable input channels
        input_channels_to_use = [
            ic
            for ic in features.keys()
            if ic in self.get_input_channels_in_graph(graph_id)
        ]

        for input_channel in input_channels_to_use:
            # Calculate sample count for each type
            existing_count, informed_count = self._sample_count(
                input_channel, features, graph_id
            )

            # Sample hypotheses based on their type
            (
                existing_possible_locations,
                existing_possible_poses,
                existing_hypotheses_evidence,
            ) = self._sample_existing(graph_id, existing_count, input_channel)
            (
                informed_possible_locations,
                informed_possible_poses,
                informed_hypotheses_feature_evidence,
            ) = self._sample_informed(features, graph_id, informed_count, input_channel)

            # We only displace existing hypotheses since the newly resampled hypotheses
            # should not be affected by the displacement from the last sensory input.
            if existing_count > 0:
                existing_possible_locations, existing_hypotheses_evidence = (
                    self._displace_hypotheses_and_compute_evidence(
                        features,
                        existing_possible_locations,
                        existing_possible_poses,
                        existing_hypotheses_evidence,
                        displacements,
                        graph_id,
                        input_channel,
                    )
                )

            # Concatenate and rebuild hypothesis space
            channel_possible_locations = np.vstack(
                [existing_possible_locations, informed_possible_locations]
            )
            channel_possible_poses = np.vstack(
                [existing_possible_poses, informed_possible_poses]
            )
            channel_hypotheses_evidence = np.hstack(
                [existing_hypotheses_evidence, informed_hypotheses_feature_evidence]
            )

            self._set_hypotheses_in_hpspace(
                graph_id=graph_id,
                input_channel=input_channel,
                new_location_hypotheses=channel_possible_locations,
                new_pose_hypotheses=channel_possible_poses,
                new_evidence=channel_hypotheses_evidence,
            )

        end_time = time.time()
        assert not np.isnan(np.max(self.evidence[graph_id])), "evidence contains NaN."
        logging.debug(
            f"evidence update for {graph_id} took "
            f"{np.round(end_time - start_time, 2)} seconds."
            f" New max evidence: {np.round(np.max(self.evidence[graph_id]), 3)}"
        )

    def _sample_count(
        self, input_channel: str, features: Dict, graph_id: str
    ) -> Tuple[int, int]:
        """Calculates the number of existing and informed hypotheses needed.

        Args:
            input_channel (str): The channel for which to calculate hypothesis count.
            features (dict): Input features containing pose information.
            graph_id (str): Identifier of the graph being queried.

        Returns:
            Tuple[int, int]: A tuple containing the number of existing and new
                hypotheses needed. Existing hypotheses are maintained from existing ones
                while new hypotheses will be initialized, informed by pose sensory
                information.

        Notes:
            This function takes into account the following ratios:
              - `hypotheses_count_multiplier`: multiplier for total count calculation.
              - `hypotheses_existing_to_new_ratio`: ratio between existing and new
                hypotheses to be sampled.
        """
        graph_num_points = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        ).shape[0]
        num_hyps_per_node = self._num_hyps_per_node(features, input_channel)
        full_informed_count = graph_num_points * num_hyps_per_node

        mapper = self.channel_hypothesis_mapping[graph_id]
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

        return (
            int(existing_maintained),
            int(new_informed),
        )

    def _sample_informed(
        self,
        features: Dict,
        graph_id: str,
        informed_count: int,
        input_channel: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        num_hyps_per_node = self._num_hyps_per_node(features, input_channel)

        # === Calculate selected evidence by top-k indices === #
        if self.use_features_for_matching[input_channel]:
            node_feature_evidence = self._calculate_feature_evidence_for_all_nodes(
                features, input_channel, graph_id
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
            sensed_directions = features[input_channel]["pose_vectors"]
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

        return selected_locations, selected_rotations, selected_feature_evidence

    def _sample_existing(
        self,
        graph_id: str,
        existing_count: int,
        input_channel: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples the specified number of existing hypotheses to retain.

        Args:
            graph_id (str): Identifier of the graph being queried.
            existing_count (int): Number of existing hypotheses to sample.
            input_channel: The channel for which hypotheses are sampled.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
                selected locations, rotations, and evidence scores.
        """
        # Return empty arrays for no hypotheses to sample
        if existing_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

        # TODO implement sampling based on evidence slope.
        mapper = self.channel_hypothesis_mapping[graph_id]
        selected_locations = mapper.extract(
            self.possible_locations[graph_id], input_channel
        )[:existing_count]
        selected_rotations = mapper.extract(
            self.possible_poses[graph_id], input_channel
        )[:existing_count]
        selected_evidence = mapper.extract(self.evidence[graph_id], input_channel)[
            :existing_count
        ]

        return selected_locations, selected_rotations, selected_evidence
