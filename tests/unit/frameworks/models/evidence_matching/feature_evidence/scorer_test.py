# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock, sentinel

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.scorer import (
    DefaultFeatureEvidenceScorer,
)

CHANNEL = "channel"
GRAPH_ID = "graph_id"


class DefaultFeatureEvidenceScorerTest(TestCase):
    @given(
        num_nodes=st.integers(min_value=1, max_value=100),
    )
    def test_returns_all_zeros_when_input_channel_is_not_used_for_matching(
        self, num_nodes: int
    ) -> None:
        mock_graph_memory = Mock()
        mock_graph_memory.get_feature_array = Mock(
            return_value={CHANNEL: np.zeros((num_nodes, 3))}
        )
        mock_selector = Mock(select=Mock(return_value={CHANNEL: False}))

        scorer = DefaultFeatureEvidenceScorer(
            graph_memory=mock_graph_memory,
            feature_weights={},
            tolerances={},
            features_for_matching_selector=mock_selector,
        )

        result = scorer(
            graph_id=GRAPH_ID,
            input_channel=CHANNEL,
            query_features={},
        )
        np.testing.assert_array_equal(result, np.zeros(num_nodes))

    def test_returns_feature_evidence_multiplied_by_feature_evidence_increment_when_input_channel_is_used_for_matching(self) -> None:  # noqa: E501
        mock_graph_memory = Mock()
        mock_graph_memory.get_feature_array = Mock(
            return_value={CHANNEL: sentinel.feature_array}
        )
        mock_graph_memory.get_feature_order = Mock(
            return_value={CHANNEL: sentinel.feature_order}
        )
        mock_selector = Mock(select=Mock(return_value={CHANNEL: True}))
        mock_calculator = Mock(calculate=Mock(return_value=np.ones(100)))
        feature_evidence_increment = 0.5
        expected_result = np.ones(100) * feature_evidence_increment

        scorer = DefaultFeatureEvidenceScorer(
            graph_memory=mock_graph_memory,
            feature_weights={
                CHANNEL: sentinel.feature_weights
            },
            tolerances={
                CHANNEL: sentinel.tolerances
            },
            features_for_matching_selector=mock_selector,
            feature_evidence_calculator=mock_calculator,
            feature_evidence_increment=feature_evidence_increment,
        )

        result = scorer(
            graph_id=GRAPH_ID,
            input_channel=CHANNEL,
            query_features=sentinel.query_features,
        )
        np.testing.assert_array_equal(result, expected_result)
        mock_calculator.calculate.assert_called_once_with(
            channel_feature_array=sentinel.feature_array,
            channel_feature_order=sentinel.feature_order,
            channel_feature_weights=sentinel.feature_weights,
            channel_query_features=sentinel.query_features,
            channel_tolerances=sentinel.tolerances,
        )
