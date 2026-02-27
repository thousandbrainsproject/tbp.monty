# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)


class SDRFeatureEvidenceCalculator:
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
        input_channel: str,
    ) -> np.ndarray:
        """Calculates feature evidence for all nodes stored in a graph.

        This calculation tests if the input_channel is a learning_module. If so,
        a different function is used for feature comparison.

        Note: This assumes that learning modules always outputs 1 feature, object_id.
        If the learning modules output more than object_id features, we need to
        compare these according to their weights.

        Returns:
            The feature evidence for all nodes.
        """
        if input_channel.startswith("learning_module"):
            return SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(  # noqa: E501
                channel_feature_array=channel_feature_array,
                channel_feature_weights=channel_feature_weights,
                channel_query_features=channel_query_features,
                channel_tolerances=channel_tolerances,
            )

        return DefaultFeatureEvidenceCalculator.calculate(
            channel_feature_array=channel_feature_array,
            channel_feature_order=channel_feature_order,
            channel_feature_weights=channel_feature_weights,
            channel_query_features=channel_query_features,
            channel_tolerances=channel_tolerances,
            input_channel=input_channel,
        )

    @staticmethod
    def calculate_feature_evidence_sdr_for_all_nodes(
        channel_feature_array: np.ndarray,
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray:
        """Calculate overlap between stored and query SDR features.

        Calculates the overlap between the SDR features stored at every location in
        the graph and the query SDR feature. This overlap is then compared to the
        tolerance value and the result is used for adjusting the evidence score.

        We use the tolerance (in overlap bits) for generalization. If two objects are
        close enough, their overlap in bits should be higher than the set tolerance
        value.

        The tolerance sets the lowest overlap for adding evidence, the range
        [tolerance, sdr_on_bits] is mapped to [0,1] evidence points. Any overlap less
        than tolerance will not add any evidence. These evidence scores are then
        multiplied by the feature weight of object_ids which scales all of the
        evidence points to the range [0, channel_feature_weights["object_id"]].

        The below variables have the following shapes:
            - channel_feature_array: (n, sdr_length)
            - channel_query_features["object_id"]: (sdr_length)
            - query_feat: (sdr_length, 1)
            - np.matmul(channel_feature_array, query_feat): (n, 1)
            - overlaps: (n)

        Returns:
            The normalized overlaps.
        """
        query_feat = np.expand_dims(channel_query_features["object_id"], 1)
        tolerance = channel_tolerances["object_id"]
        sdr_on_bits = query_feat.sum(axis=0)

        overlaps = channel_feature_array @ query_feat.squeeze(-1)
        normalized_overlaps = (overlaps - tolerance) / (sdr_on_bits - tolerance)
        normalized_overlaps[normalized_overlaps < 0] = 0.0

        normalized_overlaps *= channel_feature_weights["object_id"]
        return normalized_overlaps
