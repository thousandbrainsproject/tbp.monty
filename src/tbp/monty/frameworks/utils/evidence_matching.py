# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from typing import OrderedDict as OrderedDictType

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    ChannelHypotheses,
    Hypotheses,
)


class ChannelMapper:
    """Marks the range of hypotheses that correspond to each input channel.

    The `EvidenceGraphLM` implementation stacks the hypotheses from all input channels
    in the same array to perform efficient vector operations on them. Therefore, we
    need to keep track of which indices in the stacked array correspond to which input
    channel. This class stores only the sizes of the input channels in an ordered data
    structure (OrderedDict), and computes the range of indices for each channel. Storing
    the sizes of channels in an ordered dictionary allows us to insert or remove
    channels, as well as dynamically resize them.

    """

    def __init__(self, channel_sizes: Optional[Dict[str, int]] = None) -> None:
        """Initializes the ChannelMapper with an ordered dictionary of channel sizes.

        Args:
            channel_sizes (Optional[Dict[str, int]]): Dictionary of
                {channel_name: size}.
        """
        self.channel_sizes: OrderedDictType[str, int] = (
            OrderedDict(channel_sizes) if channel_sizes else OrderedDict()
        )

    @property
    def channels(self) -> List[str]:
        """Returns the existing channel names.

        Returns:
            List[str]: List of channel names.
        """
        return list(self.channel_sizes.keys())

    @property
    def total_size(self) -> int:
        """Returns the total number of hypotheses across all channels.

        Returns:
            int: Total size across all channels.
        """
        return sum(self.channel_sizes.values())

    def channel_size(self, channel_name: str) -> int:
        """Returns the total number of hypotheses for a specific channel.

        Returns:
            int: Size of channel

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        return self.channel_sizes[channel_name]

    def channel_range(self, channel_name: str) -> Tuple[int, int]:
        """Returns the start and end indices of the given channel.

        Args:
            channel_name (str): The name of the channel.

        Returns:
            Tuple[int, int]: The start and end indices of the channel.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        start = 0
        for name, size in self.channel_sizes.items():
            if name == channel_name:
                return (start, start + size)
            start += size

    def resize_channel_by(self, channel_name: str, value: int) -> None:
        """Increases or decreases the channel by a specific amount.

        Args:
            channel_name (str): The name of the channel.
            value (int): The value used to modify the channel size.
                Use a negative value to decrease the size.

        Raises:
            ValueError: If the channel is not found or the requested size is negative.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] + value <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] += value

    def resize_channel_to(self, channel_name: str, new_size: int) -> None:
        """Sets the size of the given channel to a specific value.

        Args:
            channel_name (str): The name of the channel.
            new_size (int): The new size to set for the channel.

        Raises:
            ValueError: If the channel is not found or if the new size is not positive.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if new_size <= 0:
            raise ValueError(f"Channel '{channel_name}' size must be positive.")
        self.channel_sizes[channel_name] = new_size

    def add_channel(
        self, channel_name: str, size: int, position: Optional[int] = None
    ) -> None:
        """Adds a new channel at a specified position (default is at the end).

        Args:
            channel_name (str): The name of the new channel.
            size (int): The size of the new channel.
            position (Optional[int]): The index at which to insert the channel.

        Raises:
            ValueError: If the channel already exists or position is out of bounds.
        """
        if channel_name in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' already exists.")

        if isinstance(position, int) and position >= len(self.channel_sizes):
            raise ValueError(f"Position index '{position}' is out of bounds.")

        if position is None:
            self.channel_sizes[channel_name] = size
        else:
            items = list(self.channel_sizes.items())
            items.insert(position, (channel_name, size))
            self.channel_sizes = OrderedDict(items)

    def extract(self, original: np.ndarray, channel: str) -> np.ndarray:
        """Extracts the portion of the original array corresponding to a given channel.

        Args:
            original (np.ndarray): The full hypotheses array across all channels.
            channel (str): The name of the channel to extract.

        Returns:
            np.ndarray: The extracted slice of the original array. Returns a view, not
                a copy of the original array.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel not in self.channel_sizes:
            raise ValueError(f"Channel '{channel}' not found.")

        start, end = self.channel_range(channel)
        return original[start:end]

    def extract_hypotheses(
        self, hypotheses: Hypotheses, channel: str
    ) -> ChannelHypotheses:
        """Extracts the hypotheses corresponding to a given channel.

        Args:
            hypotheses (Hypotheses): The full hypotheses array across all channels.
            channel (str): The name of the channel to extract.

        Returns:
            ChannelHypotheses: The hypotheses corresponding to the given channel.
        """
        return ChannelHypotheses(
            input_channel=channel,
            evidence=self.extract(hypotheses.evidence, channel),
            locations=self.extract(hypotheses.locations, channel),
            poses=self.extract(hypotheses.poses, channel),
        )

    def update(
        self, original: np.ndarray, channel: str, data: np.ndarray
    ) -> np.ndarray:
        """Inserts data into the original array at the position of the given channel.

        This function inserts new data at the index range previously associated with
        the provided channel. If the new data is of the same shape as the existing
        channel data shape, we simply replace the data at the channel range indices.
        Otherwise, We split the original array around the input channel range, then
        concatenate the before and after splits with the data to be inserted. This
        accommodates 'data' being of a different size than the current channel size.

        For example, if original has the shape (20, 3), channel start index is 10,
        channel end index is 13, and the data has the shape (5, 3). We would concatenate
        as such: (original[0:10], data, original[13:]). This will result in an array of
        the shape (22, 3), i.e., we removed 3 rows and added new 5 rows.

        Args:
            original (np.ndarray): The original array.
            channel (str): The name of the input channel.
            data (np.ndarray): The new data to insert.

        Returns:
            np.ndarray: The resulting array after insertion. Can return a new copy or a
                view, depending on whether the inserted data is of the same size as the
                existing channel.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel not in self.channel_sizes:
            raise ValueError(f"Channel '{channel}' not found.")

        start, end = self.channel_range(channel)

        if self.channel_sizes[channel] == data.shape[0]:
            # returns a view not a copy
            original[start:end] = data
        else:
            # returns a copy not a view
            original = np.concatenate([original[:start], data, original[end:]], axis=0)

        return original

    def __repr__(self) -> str:
        """Returns a string representation of the current channel mapping.

        Returns:
            str: String representation of the channel mappings.
        """
        ranges = {ch: self.channel_range(ch) for ch in self.channel_sizes}
        return f"ChannelMapper({ranges})"


class EvidenceSlopeTracker:
    """Tracks the slopes of evidence streams over a sliding window per input channel.

    Each input channel maintains its own hypotheses with independent evidence histories.
    This tracker supports adding, updating, pruning, and analyzing hypotheses per
    channel.

    Attributes:
        window_size (int): Number of past values to consider for slope calculation.
        min_age (int): Minimum number of updates before a hypothesis can be considered
            for removal.
        data (Dict[str, np.ndarray]): Maps channel names to their hypothesis evidence
            buffers.
        age (Dict[str, np.ndarray]): Maps channel names to hypothesis age counters.
    """

    def __init__(self, window_size: int = 3, min_age: int = 2) -> None:
        """Initializes the EvidenceSlopeTracker.

        Args:
            window_size (int, optional): Number of evidence points per hypothesis.
            min_age (int, optional): Minimum number of updates before removal is
                allowed.
        """
        self.window_size = window_size
        self.min_age = min_age
        self.data: Dict[str, np.ndarray] = {}
        self.age: Dict[str, np.ndarray] = {}

    def total_size(self, channel: str) -> int:
        """Returns the number of hypotheses in a given channel.

        Args:
            channel (str): Name of the input channel.

        Returns:
            int: Number of hypotheses currently tracked in the channel.
        """
        return self.data.get(channel, np.empty((0, self.window_size))).shape[0]

    def valid_indices_mask(self, channel: str) -> np.ndarray:
        """Returns a boolean mask for hypotheses valid for removal in a channel.

        Args:
            channel (str): Name of the input channel.

        Returns:
            np.ndarray: Boolean array indicating valid hypotheses (age >= min_age).
        """
        return self.age[channel] >= self.min_age

    def must_keep_mask(self, channel: str) -> np.ndarray:
        """Returns a boolean mask for hypotheses that must be kept due to age.

        Args:
            channel (str): Name of the input channel.

        Returns:
            np.ndarray: Boolean array indicating hypotheses that must be kept
                (age < min_age).
        """
        return self.age[channel] < self.min_age

    def add_hyp(self, num_new_hyp: int, channel: str) -> None:
        """Adds new hypotheses to the specified input channel.

        Args:
            num_new_hyp (int): Number of new hypotheses to add.
            channel (str): Name of the input channel.
        """
        new_data = np.full((num_new_hyp, self.window_size), np.nan)
        new_age = np.zeros(num_new_hyp, dtype=int)

        if channel not in self.data:
            self.data[channel] = new_data
            self.age[channel] = new_age
        else:
            self.data[channel] = np.vstack((self.data[channel], new_data))
            self.age[channel] = np.concatenate((self.age[channel], new_age))

    def update(self, values: List[float] | np.ndarray, channel: str) -> None:
        """Updates all hypotheses in a channel with new evidence values.

        Args:
            values (List[float] | np.ndarray): List or array of new evidence values.
            channel (str): Name of the input channel.

        Raises:
            ValueError: If the channel doesn't exist or the number of values is
                incorrect.
        """
        values = np.array(values, dtype=float)
        if channel not in self.data:
            raise ValueError(f"Channel '{channel}' does not exist.")

        if values.shape[0] != self.total_size(channel):
            raise ValueError(
                f"Expected {self.total_size(channel)} values, but got {len(values)}"
            )

        # Shift evidence data by one step
        self.data[channel][:, :-1] = self.data[channel][:, 1:]

        # Add new evidence data
        self.data[channel][:, -1] = values

        # Increment age
        self.age[channel] += 1

    def _calculate_slopes(self, channel: str) -> np.ndarray:
        """Computes the average slope of hypotheses in a channel.

        Args:
            channel (str): Name of the input channel.

        Returns:
            np.ndarray: Array of average slopes, one per hypothesis.
        """
        diffs = np.diff(self.data[channel], axis=1)
        valid_steps = np.count_nonzero(~np.isnan(diffs), axis=1)
        valid_steps = np.where(valid_steps == 0, np.nan, valid_steps)
        return np.nansum(diffs, axis=1) / valid_steps

    def remove_hyp(self, hyp_ids: List[int], channel: str) -> None:
        """Removes specific hypotheses by index in the specified channel.

        Args:
            hyp_ids (List[int]): List of hypothesis indices to remove.
            channel (str): Name of the input channel.
        """
        mask = np.ones(self.total_size(channel), dtype=bool)
        mask[hyp_ids] = False
        self.data[channel] = self.data[channel][mask]
        self.age[channel] = self.age[channel][mask]

    def calculate_keep_and_remove_ids(
        self, desired_keep: int, channel: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determines which hypotheses to keep and which to remove in a channel.

        Hypotheses with the lowest average slope are selected for removal.

        Args:
            desired_keep (int): Target number of hypotheses to retain.
            channel (str): Name of the input channel.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - to_keep: Indices of hypotheses to retain.
                - to_remove: Indices of hypotheses to remove.

        Raises:
            ValueError: If the channel does not exist.
        """
        if channel not in self.data:
            raise ValueError(f"Channel '{channel}' does not exist.")

        valid_mask = self.valid_indices_mask(channel)
        slopes = self._calculate_slopes(channel)
        valid_slopes = slopes[valid_mask]
        total_ids = np.arange(self.total_size(channel))
        valid_ids = total_ids[valid_mask]

        sorted_indices = np.argsort(valid_slopes)
        to_remove = valid_ids[sorted_indices[:-desired_keep]]
        to_keep = np.setdiff1d(valid_ids, to_remove)
        return to_keep, to_remove
