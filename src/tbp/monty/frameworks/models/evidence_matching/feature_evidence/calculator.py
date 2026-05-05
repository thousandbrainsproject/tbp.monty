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

_SKIP_FEATURES = frozenset({"pose_vectors", "pose_fully_defined"})
_CIRCULAR_FEATURES = frozenset({"hsv"})
_CATEGORICAL_FEATURES = frozenset({"object_id"})
_CIRCULAR_RANGE = 1


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

        Each feature column is classified into one of three kinds and compared
        accordingly:

        - numeric: `|stored - observed|`
        - circular (e.g. HSV hue): the smallest wrap-around distance on
          `[0, _CIRCULAR_RANGE)`
        - categorical (e.g. ``object_id``): `0` if equal, `1` otherwise

        The resulting per-column difference is mapped to evidence in `[0, 1]`
        via `clip(tolerance - difference, 0, inf) / tolerance`: an evidence
        of 1 is a perfect match, 0 once the difference reaches the tolerance.
        Per-channel evidence is the feature-weighted average across columns.

        If a node does not store a given feature, evidence will be nan.

        `input_channel` indicates where the sensed features are coming from
        and thereby tells this function to which features in the graph they
        need to be compared.

        Returns:
            The feature evidence for all nodes.
        """
        n_cols = channel_feature_array.shape[1]
        tolerance_list = np.full(n_cols, np.nan)
        feature_weight_list = np.full(n_cols, np.nan)
        feature_list = np.full(n_cols, np.nan)
        numeric_var = np.zeros(n_cols, dtype=bool)
        circular_var = np.zeros(n_cols, dtype=bool)
        categorical_var = np.zeros(n_cols, dtype=bool)

        start_idx = 0
        for feature in channel_feature_order:
            if feature in _SKIP_FEATURES:
                continue
            if hasattr(channel_query_features[feature], "__len__"):
                feature_length = len(channel_query_features[feature])
            else:
                feature_length = 1
            end_idx = start_idx + feature_length
            feature_list[start_idx:end_idx] = channel_query_features[feature]
            tolerance_list[start_idx:end_idx] = channel_tolerances[feature]
            feature_weight_list[start_idx:end_idx] = channel_feature_weights[feature]

            if feature == "hsv":
                # H is circular, S and V are numeric
                circular_var[start_idx] = True
                numeric_var[start_idx + 1 : end_idx] = True
            elif feature in _CATEGORICAL_FEATURES:
                categorical_var[start_idx:end_idx] = True
            else:
                numeric_var[start_idx:end_idx] = True

            start_idx = end_idx

        assert (numeric_var ^ circular_var ^ categorical_var).all(), (
            "feature kind masks must be mutually exclusive and exhaustive"
        )

        feature_differences = np.zeros_like(channel_feature_array)
        feature_differences[:, numeric_var] = np.abs(
            channel_feature_array[:, numeric_var] - feature_list[numeric_var]
        )
        cnode_fs = channel_feature_array[:, circular_var]
        cquery_fs = feature_list[circular_var]
        feature_differences[:, circular_var] = np.min(
            [
                np.abs(_CIRCULAR_RANGE + cnode_fs - cquery_fs),
                np.abs(cnode_fs - cquery_fs),
                np.abs(cnode_fs - (cquery_fs + _CIRCULAR_RANGE)),
            ],
            axis=0,
        )
        feature_differences[:, categorical_var] = (
            channel_feature_array[:, categorical_var] != feature_list[categorical_var]
        ).astype(channel_feature_array.dtype)

        # any difference < tolerance should be positive evidence
        # any difference >= tolerance should be 0 evidence
        feature_evidence = np.clip(tolerance_list - feature_differences, 0, np.inf)
        # normalize evidence to be in [0, 1]
        feature_evidence = feature_evidence / tolerance_list
        return np.average(feature_evidence, weights=feature_weight_list, axis=1)
