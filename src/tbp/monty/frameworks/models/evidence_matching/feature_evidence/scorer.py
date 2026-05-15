# Copyright 2026 Thousand Brains Project
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
import numpy.typing as npt

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


class FeatureEvidenceScorer(Protocol):
    """A scorer that calculates the feature evidence for a given channel."""

    def __call__(
        self,
        graph_id: str,
        input_channel: str,
        query_features: dict,
    ) -> npt.NDArray[np.float64]:
        """Calculate the feature evidence for a given channel.

        Args:
            graph_id: The graph id.
            input_channel: The channel.
            query_features: The query features.

        Returns:
            The feature evidence for the given channel.
        """
        ...


class DefaultFeatureEvidenceScorer(FeatureEvidenceScorer):
    """Default scorer that calculates the feature evidence for a given channel.

    The feature evidence is a score between 0 and 1 that indicates how well the observed
    features match the stored features. If  the channel is used for matching, the score
    is calculated by the feature evidence calculator and multiplied by the feature
    evidence increment. Otherwise, all zeros are returned.
    """

    def __init__(
        self,
        graph_memory: EvidenceGraphMemory,
        feature_weights: dict,
        tolerances: dict,
        feature_evidence_calculator: type[
            FeatureEvidenceCalculator
        ] = DefaultFeatureEvidenceCalculator,
        feature_evidence_increment: int = 1,
        features_for_matching_selector: type[
            FeaturesForMatchingSelector
        ] = DefaultFeaturesForMatchingSelector,
    ) -> None:
        """Initializes the DefaultFeatureEvidenceScorer.

        Args:
            graph_memory: The graph memory to read graphs from.
            feature_weights: How much should each feature be weighted when
                calculating the evidence update for hypothesis. Weights are stored in a
                dictionary with keys corresponding to features (same as keys in
                tolerances).
            tolerances: How much can each observed feature deviate from the
                stored features to still be considered a match.
            feature_evidence_calculator: Calculator that calculates feature evidence for
                all nodes.
            feature_evidence_increment: Feature evidence (between 0 and 1) is
                multiplied by this value before being added to the overall evidence of
                a hypothesis. This factor is only multiplied with the feature evidence
                (not the pose evidence as opposed to the present_weight). Defaults to 1.
            features_for_matching_selector: Selector that selects if features should be
                used for matching. Defaults to the default selector.
        """
        self._use_features_for_matching_by_channel = (
            features_for_matching_selector.select(
                feature_evidence_increment=feature_evidence_increment,
                feature_weights=feature_weights,
                tolerances=tolerances,
            )
        )
        self._graph_memory = graph_memory
        self._feature_weights = feature_weights
        self._tolerances = tolerances
        self._feature_evidence_calculator = feature_evidence_calculator
        self._feature_evidence_increment = feature_evidence_increment

    def _use_features_for_matching(self, input_channel: str) -> bool:
        return self._use_features_for_matching_by_channel[input_channel]

    def __call__(
        self,
        graph_id: str,
        input_channel: str,
        query_features: dict,
    ) -> npt.NDArray[np.float64]:
        if not self._use_features_for_matching(input_channel):
            n_nodes = self._graph_memory.get_feature_array(graph_id)[
                input_channel
            ].shape[0]
            return np.zeros(n_nodes, dtype=np.float64)

        evidence = self._feature_evidence_calculator.calculate(
            channel_feature_array=self._graph_memory.get_feature_array(graph_id)[
                input_channel
            ],
            channel_feature_order=self._graph_memory.get_feature_order(graph_id)[
                input_channel
            ],
            channel_feature_weights=self._feature_weights[input_channel],
            channel_query_features=query_features,
            channel_tolerances=self._tolerances[input_channel],
        )
        return evidence * self._feature_evidence_increment
