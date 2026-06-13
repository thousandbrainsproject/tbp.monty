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

import cv2
import numpy as np


class FeatureEvidenceCalculator(Protocol):
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray: ...


class DefaultFeatureEvidenceCalculator:
    SKIP_FEATURES = frozenset({"pose_vectors", "pose_fully_defined"})
    CIRCULAR_FEATURES = frozenset({"hsv"})
    CATEGORICAL_FEATURES = frozenset({"object_id"})
    HISTOGRAM_FEATURES = frozenset({"ltp"})
    CIRCULAR_RANGE = 1

    @classmethod
    def calculate(
        cls,
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray:
        """Calculate the feature evidence for all nodes stored in a graph.

        For each node, compares the stored features against the observed
        query features and returns a score in `[0, 1]`: 1 for a perfect
        match, decaying to 0 once the difference exceeds the per-feature
        tolerance. Nodes with missing stored values for a feature receive
        NaN evidence.

        Args:
            channel_feature_array: Stored features for every node in the
                graph, shape `(n_nodes, n_columns)`. Columns follow the
                layout given by `channel_feature_order`.
            channel_feature_order: Feature names in the order they appear
                across the columns of `channel_feature_array`.
            channel_feature_weights: Per-feature weights used to combine
                per-column evidence into a single per-node score.
            channel_query_features: Observed feature values to compare
                against the stored features, keyed by feature name.
            channel_tolerances: Per-feature tolerance, the largest
                difference that still produces non-zero evidence.

        Returns:
            The feature evidence for all nodes, shape `(n_nodes,)`.
        """
        n_cols = channel_feature_array.shape[1]
        tolerance_list = np.full(n_cols, np.nan)
        feature_weight_list = np.full(n_cols, np.nan)
        feature_list = np.full(n_cols, np.nan)
        numeric_var = np.zeros(n_cols, dtype=bool)
        circular_var = np.zeros(n_cols, dtype=bool)
        categorical_var = np.zeros(n_cols, dtype=bool)
        histogram_var = np.zeros(n_cols, dtype=bool)

        # Column ranges (start, end) occupied by each histogram feature, so that the
        # chi-square distance can be computed per histogram rather than across all
        # columns lumped together.
        histogram_slices: list[tuple[int, int]] = []

        start_idx = 0
        for feature in channel_feature_order:
            if feature in cls.SKIP_FEATURES:
                continue
            if hasattr(channel_query_features[feature], "__len__"):
                feature_length = len(channel_query_features[feature])
            else:
                feature_length = 1
            end_idx = start_idx + feature_length
            feature_list[start_idx:end_idx] = channel_query_features[feature]
            tolerance_list[start_idx:end_idx] = channel_tolerances[feature]

            if feature in cls.CIRCULAR_FEATURES:
                # H is circular, S and V are numeric
                circular_var[start_idx] = True
                numeric_var[start_idx + 1 : end_idx] = True
                feature_weight_list[start_idx:end_idx] = channel_feature_weights[
                    feature
                ]
            elif feature in cls.CATEGORICAL_FEATURES:
                categorical_var[start_idx:end_idx] = True
                feature_weight_list[start_idx:end_idx] = channel_feature_weights[
                    feature
                ]
            elif feature in cls.HISTOGRAM_FEATURES:
                histogram_var[start_idx:end_idx] = True
                histogram_slices.append((start_idx, end_idx))
                # A histogram occupies one column per bin but is conceptually a
                # single feature. Each bin column receives the same per-node
                # chi-square distance, so we spread the feature weight evenly across
                # the bins. This keeps the histogram's contribution to the weighted
                # average equal to `channel_feature_weights[feature]`, independent of
                # the number of bins.
                feature_weight_list[start_idx:end_idx] = (
                    channel_feature_weights[feature] / feature_length
                )
            else:
                numeric_var[start_idx:end_idx] = True
                feature_weight_list[start_idx:end_idx] = channel_feature_weights[
                    feature
                ]

            start_idx = end_idx

        assert (numeric_var ^ circular_var ^ categorical_var ^ histogram_var).all(), (
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
                np.abs(cls.CIRCULAR_RANGE + cnode_fs - cquery_fs),
                np.abs(cnode_fs - cquery_fs),
                np.abs(cnode_fs - (cquery_fs + cls.CIRCULAR_RANGE)),
            ],
            axis=0,
        )
        feature_differences[:, categorical_var] = (
            channel_feature_array[:, categorical_var] != feature_list[categorical_var]
        ).astype(channel_feature_array.dtype)
        # Use Chi-2 distance for histogram features. The distance is computed per
        # node between the stored histogram and the query histogram, and broadcast
        # across the histogram's bin columns (each bin shares the same distance and a
        # correspondingly reduced weight). cv2.compareHist requires single-precision
        # float input, hence the explicit cast.
        for hist_start, hist_end in histogram_slices:
            query_hist = feature_list[hist_start:hist_end].astype(np.float32)
            stored_hists = channel_feature_array[:, hist_start:hist_end].astype(
                np.float32
            )
            chi_distances = np.array(
                [
                    cv2.compareHist(
                        stored_hist.astype(np.float32),
                        query_hist.astype(np.float32),
                        cv2.HISTCMP_CHISQR,
                    )
                    for stored_hist in stored_hists
                ],
                dtype=channel_feature_array.dtype,
            )
            feature_differences[:, hist_start:hist_end] = chi_distances[:, np.newaxis]
        # any difference < tolerance should be positive evidence
        # any difference >= tolerance should be 0 evidence
        feature_evidence = np.clip(tolerance_list - feature_differences, 0, np.inf)
        # normalize evidence to be in [0, 1]
        feature_evidence = feature_evidence / tolerance_list
        return np.average(feature_evidence, weights=feature_weight_list, axis=1)
