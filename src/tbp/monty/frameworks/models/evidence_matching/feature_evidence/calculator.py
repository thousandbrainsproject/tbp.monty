# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Protocol

import numpy as np


class FeatureEvidenceCalculator(Protocol):
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
        input_channel: str,
    ) -> np.ndarray: ...


class DefaultFeatureEvidenceCalculator:
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
        input_channel: str,  # noqa: ARG004
    ) -> np.ndarray:
        """Calculate the feature evidence for all nodes stored in a graph.

        Evidence for each feature depends on the difference between observed and stored
        features, feature weights, and distance weights.

        Evidence is a float between 0 and 1. An evidence of 1 is a perfect match; the
        larger the difference between observed and sensed features, the closer to 0
        the evidence becomes. Evidence is 0 if the difference is >= the tolerance for
        this feature.

        If a node does not store a given feature, evidence will be nan.

        input_channel indicates where the sensed features are coming from and thereby
        tells this function to which features in the graph they need to be compared.

        Returns:
            The feature evidence for all nodes.
        """
        # generate the lists of features, tolerances, and whether features are circular
        shape_to_use = channel_feature_array.shape[1]
        tolerance_list = np.zeros(shape_to_use) * np.nan
        feature_weight_list = np.zeros(shape_to_use) * np.nan
        feature_list = np.zeros(shape_to_use) * np.nan
        circular_var = np.zeros(shape_to_use, dtype=bool)
        start_idx = 0
        for feature in channel_feature_order:
            if feature in [
                "pose_vectors",
                "pose_fully_defined",
            ]:
                continue
            if hasattr(channel_query_features[feature], "__len__"):
                feature_length = len(channel_query_features[feature])
            else:
                feature_length = 1
            end_idx = start_idx + feature_length
            feature_list[start_idx:end_idx] = channel_query_features[feature]
            tolerance_list[start_idx:end_idx] = channel_tolerances[feature]
            feature_weight_list[start_idx:end_idx] = channel_feature_weights[feature]
            circular_var[start_idx:end_idx] = (
                [True, False, False] if feature == "hsv" else False
            )
            circ_range = 1
            start_idx = end_idx

        feature_differences = np.zeros_like(channel_feature_array)
        feature_differences[:, ~circular_var] = np.abs(
            channel_feature_array[:, ~circular_var] - feature_list[~circular_var]
        )
        cnode_fs = channel_feature_array[:, circular_var]
        cquery_fs = feature_list[circular_var]
        feature_differences[:, circular_var] = np.min(
            [
                np.abs(circ_range + cnode_fs - cquery_fs),
                np.abs(cnode_fs - cquery_fs),
                np.abs(cnode_fs - (cquery_fs + circ_range)),
            ],
            axis=0,
        )
        # any difference < tolerance should be positive evidence
        # any difference >= tolerance should be 0 evidence
        feature_evidence = np.clip(tolerance_list - feature_differences, 0, np.inf)
        # normalize evidence to be in [0, 1]
        feature_evidence = feature_evidence / tolerance_list
        return np.average(feature_evidence, weights=feature_weight_list, axis=1)
