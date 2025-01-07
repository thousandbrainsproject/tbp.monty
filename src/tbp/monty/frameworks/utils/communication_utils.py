# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


def get_state_from_channel(states, channel_name):
    """Given a list of states, return the state of the specified channel.

    Args:
        states: List of states
        channel_name: The name of the channel to return the state for

    Returns:
        The state of the specified channel

    Raises:
        ValueError: If the channel name is not found in the states
    """
    for state in states:
        if state.sender_id == channel_name:
            return state

    raise ValueError(f"Channel {channel_name} not found in states")


def get_first_sensory_state(states):
    """Given a list of states return the first one from a sensory channel.

    Returns:
        First state from a sensory channel.
    """
    for state in states:
        if state.sender_type == "SM":
            return state
    return None
