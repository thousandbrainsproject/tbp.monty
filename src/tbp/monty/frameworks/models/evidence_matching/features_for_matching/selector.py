# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Protocol


class FeaturesForMatchingSelector(Protocol):
    @staticmethod
    def select(
        feature_evidence_increment: int,
        feature_weights: dict,
        tolerances: dict,
    ) -> dict[str, bool]: ...


class DefaultFeaturesForMatchingSelector:
    @staticmethod
    def select(
        feature_evidence_increment: int,
        feature_weights: dict,
        tolerances: dict,
    ) -> dict[str, bool]:
        use_features = {}
        for input_channel in tolerances.keys():
            if (
                input_channel not in feature_weights.keys()
                or feature_evidence_increment <= 0
            ):
                use_features[input_channel] = False
            else:
                non_morphological_features = {
                    k
                    for k in feature_weights[input_channel].keys()
                    if k not in ["pose_vectors", "pose_fully_defined"]
                }
                feature_weights_provided = len(non_morphological_features) > 0
                use_features[input_channel] = feature_weights_provided
        return use_features
