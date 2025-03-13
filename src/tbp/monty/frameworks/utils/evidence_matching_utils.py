# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections import OrderedDict


class ChannelMapper:
    """This class marks the range of hypotheses that correspond to each input channel.

    Instead of storing the actual range as a List or Tuple of indices,
    we store the number of hypotheses per input channel and calculate
    the range in a cumulative sum manner as shown in the `get_channel_range`
    function.

    This allows us to dynamically change the range of one channel without
    having to recompute the range of all subsequent channels. We also do not
    need to store "num_hypotheses" as it can be computed with the `total_size`
    property.
    """

    def __init__(self, channel_sizes=None):
        """Initialize the ChannelMapper with an ordered dictionary of channel sizes.

        :param channel_sizes: Dict of {channel_name: size}, maintaining order.
        """
        self.channel_sizes = (
            OrderedDict(channel_sizes) if channel_sizes else OrderedDict()
        )

    @property
    def channels(self):
        """Returns the existing channel names."""
        return list(self.channel_sizes.keys())

    @property
    def dict(self):
        """Returns the ordered dictionary."""
        return self.channel_sizes

    @property
    def total_size(self):
        """Returns the total number of hypotheses across all channels."""
        return sum(self.channel_sizes.values())

    def get_channel_range(self, channel_name):
        """Returns the start and end indices of the given channel.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        start = 0
        for name, size in self.channel_sizes.items():
            if name == channel_name:
                return (start, start + size - 1)
            start += size

    def increase_channel_size(self, channel_name, value):
        """Increases the size of the specified channel.

        :param value: int, value added or subtracted from channel size.

        Raises:
            ValueError: If the channel is not found or requested size is negative.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] + value <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] += value

    def set_channel_size(self, channel_name, value):
        """Set the size of the specified channel.

        :param value: int, value to set channel size.

        Raises:
            ValueError: If the channel is not found or requested size is negative.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] = value

    def add_channel(self, channel_name, size, position=None):
        """Adds a new channel at a specified position (default is at the end).

        Raises:
            ValueError: If the channel already exists.
        """
        if channel_name in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' already exists.")

        if position is None or position >= len(self.channel_sizes):
            self.channel_sizes[channel_name] = size
        else:
            items = list(self.channel_sizes.items())
            items.insert(position, (channel_name, size))
            self.channel_sizes = OrderedDict(items)

    def remove_channel(self, channel_name):
        """Removes a channel from the mapping.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        del self.channel_sizes[channel_name]

    def __repr__(self):
        """Return a string representation of the current channel mapping."""
        ranges = {ch: self.get_channel_range(ch) for ch in self.channel_sizes}
        return f"ChannelMapper({ranges})"
