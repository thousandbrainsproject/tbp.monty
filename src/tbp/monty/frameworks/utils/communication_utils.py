# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from tbp.monty.cmp import Message


def get_percept_from_channel(percepts: list[Message], channel_name: str) -> Message:
    """Given a list of percepts, return the percept of the specified channel.

    Args:
        percepts: List of percepts
        channel_name: The name of the channel to return the percept for

    Returns:
        The percept of the specified channel

    Raises:
        ValueError: If the channel name is not found in the percepts
    """
    for percept in percepts:
        if percept.sender_id == channel_name:
            return percept

    raise ValueError(f"Channel {channel_name} not found in percepts")
