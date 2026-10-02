# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from enum import Enum
from typing import Mapping


class PoseKind(Enum):
    """What the pose_vectors of an input channel represent."""

    SURFACE = "surface"
    """[surface_normal, principal_curvature_dir 1, principal_curvature_dir_2]
    - Surface Normals are signed, curvature directions are unsigned
    """
    OBJECT = "object"
    """Full, signed orientation of a child object."""


_POSE_KIND_BY_SENDER_TYPE = {"SM": PoseKind.SURFACE, "LM": PoseKind.OBJECT}


def channel_pose_kinds(channel_sender_types: Mapping[str, str]) -> dict[str, PoseKind]:
    """Map each input channel to the kind of pose it sends.

    Args:
        channel_sender_types: Sender type ("SM" or "LM") per input channel.

    Returns:
        Pose kind per input channel.
    """
    return {
        channel: _POSE_KIND_BY_SENDER_TYPE[sender_type]
        for channel, sender_type in channel_sender_types.items()
    }


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
