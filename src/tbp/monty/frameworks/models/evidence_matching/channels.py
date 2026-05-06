# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations


def all_usable_input_channels(
    features: dict, all_input_channels: list[str]
) -> list[str]:
    """Determine all usable input channels.

    NOTE: We might also want to check the confidence in the input-channel
    features, but this information is currently not available here.
    TODO S: Once we pull the observation class into the LM we could add this.

    Args:
        features: Input features.
        all_input_channels: All input channels that are stored in the graph.

    Returns:
        All input channels that are usable for matching.
    """
    return [ic for ic in features if ic in all_input_channels]
