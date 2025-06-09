# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations


class AllFeaturesForMatchingSelector:
    @staticmethod
    def select(
        feature_evidence_increment: int,
        feature_weights: dict,
        tolerances: dict,
    ) -> dict[str, bool]:
        """Select which features should be used for matching.

        Use this selector when you want to use all features for matching.

        Returns:
            A dictionary indicating whether to use features for each input channel.
        """
        use_features = {}
        for input_channel in tolerances.keys():
            if input_channel not in feature_weights.keys():
                use_features[input_channel] = False
            elif feature_evidence_increment <= 0:
                use_features[input_channel] = False
            else:
                use_features[input_channel] = True
        return use_features
