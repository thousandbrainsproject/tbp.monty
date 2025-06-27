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
            channel_sizes: Dictionary of {channel_name: size}.
        """
        self.channel_sizes: OrderedDictType[str, int] = (
            OrderedDict(channel_sizes) if channel_sizes else OrderedDict()
        )

    @property
    def channels(self) -> List[str]:
        """Returns the existing channel names.

        Returns:
            List of channel names.
        """
        return list(self.channel_sizes.keys())

    @property
    def total_size(self) -> int:
        """Returns the total number of hypotheses across all channels.

        Returns:
            Total size across all channels.
        """
        return sum(self.channel_sizes.values())

    def channel_size(self, channel_name: str) -> int:
        """Returns the total number of hypotheses for a specific channel.

        Returns:
            Size of channel

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        return self.channel_sizes[channel_name]

    def channel_range(self, channel_name: str) -> Tuple[int, int]:
        """Returns the start and end indices of the given channel.

        Args:
            channel_name: The name of the channel.

        Returns:
            The start and end indices of the channel.

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
            channel_name: The name of the channel.
            value: The value used to modify the channel size.
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
            channel_name: The name of the channel.
            new_size: The new size to set for the channel.

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
            channel_name: The name of the new channel.
            size: The size of the new channel.
            position: The index at which to insert the channel.

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
            original: The full hypotheses array across all channels.
            channel: The name of the channel to extract.

        Returns:
            The extracted slice of the original array. Returns a view, not a copy of the
            original array.

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
            hypotheses: The full hypotheses array across all channels.
            channel: The name of the channel to extract.

        Returns:
            The hypotheses corresponding to the given channel.
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
            original: The original array.
            channel: The name of the input channel.
            data: The new data to insert.

        Returns:
            The resulting array after insertion. Can return a new copy or a view,
            depending on whether the inserted data is of the same size as the existing
            channel.

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
            String representation of the channel mappings.
        """
        ranges = {ch: self.channel_range(ch) for ch in self.channel_sizes}
        return f"ChannelMapper({ranges})"


def evidence_update_threshold(
    evidence_threshold_config: float | str,
    x_percent_threshold: float | str,
    max_global_evidence: float,
    evidence_all_channels: np.ndarray,
) -> float:
    """Determine how much evidence a hypothesis should have to be updated.

    Args:
        evidence_threshold_config: The heuristic for deciding which
            hypotheses should be updated.
        x_percent_threshold: The x_percent value to use for deciding
            on the `evidence_update_threshold` when the `x_percent_threshold` is
            used as a heuristic.
        max_global_evidence: Highest evidence of all hypotheses (i.e.,
            current mlh evidence),
        evidence_all_channels: Evidence values for all hypotheses.

    Returns:
        The evidence update threshold.

    Raises:
        InvalidEvidenceThresholdConfig: If `evidence_threshold_config` is
            not in the allowed values
    """
    # return 0 for the threshold if there are no evidence scores
    if evidence_all_channels.size == 0:
        return 0

    if type(evidence_threshold_config) in [int, float]:
        return evidence_threshold_config
    elif evidence_threshold_config == "mean":
        return np.mean(evidence_all_channels)
    elif evidence_threshold_config == "median":
        return np.median(evidence_all_channels)
    elif isinstance(
        evidence_threshold_config, str
    ) and evidence_threshold_config.endswith("%"):
        percentage_str = evidence_threshold_config.strip("%")
        percentage = float(percentage_str)
        assert percentage >= 0 and percentage <= 100, (
            "Percentage must be between 0 and 100"
        )
        x_percent_of_max = max_global_evidence * (percentage / 100)
        return max_global_evidence - x_percent_of_max
    elif evidence_threshold_config == "x_percent_threshold":
        x_percent_of_max = max_global_evidence / 100 * x_percent_threshold
        return max_global_evidence - x_percent_of_max
    elif evidence_threshold_config == "all":
        return np.min(evidence_all_channels)
    else:
        raise InvalidEvidenceThresholdConfig(
            "evidence_threshold_config not in "
            "[int, float, '[int]%', 'mean', 'median', 'all', 'x_percent_threshold']"
        )


class InvalidEvidenceThresholdConfig(ValueError):
    """Raised when the evidence update threshold is invalid."""

    pass
