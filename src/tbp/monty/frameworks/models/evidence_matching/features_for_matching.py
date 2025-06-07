# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


def check_use_features_for_matching(
    tolerances: dict, feature_weights: dict, feature_evidence_increment: int
):
    use_features = {}
    for input_channel in tolerances.keys():
        if input_channel not in feature_weights.keys():
            use_features[input_channel] = False
        elif feature_evidence_increment <= 0:
            use_features[input_channel] = False
        else:
            feature_weights_provided = len(feature_weights[input_channel].keys()) > 2
            use_features[input_channel] = feature_weights_provided
    return use_features
