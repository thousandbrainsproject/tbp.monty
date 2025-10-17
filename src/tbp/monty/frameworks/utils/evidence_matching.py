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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from typing import OrderedDict as OrderedDictType

import numpy as np
import numpy.typing as npt

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


class EvidenceSlopeTracker:
    """Tracks the slopes of evidence streams over a sliding window per input channel.

    Each input channel maintains its own hypotheses with independent evidence histories.
    This tracker supports adding, updating, pruning, and analyzing hypotheses per
    channel.

    Note:
        - One optimization might be to treat the array of tracked values as a ring-like
            structure. Rather than shifting the values every time they are updated, we
            could just iterate an index which determines where in the ring we are. Then
            we would update one column, which based on the index, corresponds to the
            most recent values.
        - Another optimization is only track slopes not the actual evidence values. The
            pairwise slopes for previous scores are not expected to change over time
            and therefore can be calculated a single time and stored.
        - We can also test returning a random subsample of indices with
            slopes < mean(slopes) for `to_remove` instead of using `np.argsort`.

    Attributes:
        window_size: Number of past values to consider for slope calculation.
        min_age: Minimum number of updates before a hypothesis can be considered for
            removal.
        evidence_buffer: Maps channel names to their hypothesis evidence buffers.
        hyp_age: Maps channel names to hypothesis age counters.
    """

    def __init__(self, window_size: int = 10, min_age: int = 5) -> None:
        """Initializes the EvidenceSlopeTracker.

        Args:
            window_size: Number of evidence points per hypothesis.
            min_age: Minimum number of updates before removal is allowed.
        """
        self.window_size = window_size
        self.min_age = min_age
        self.evidence_buffer: dict[str, npt.NDArray[np.float64]] = {}
        self.hyp_age: dict[str, npt.NDArray[np.int_]] = {}

    def total_size(self, channel: str) -> int:
        """Returns the number of hypotheses in a given channel.

        Args:
            channel: Name of the input channel.

        Returns:
            Number of hypotheses currently tracked in the channel.
        """
        return self.evidence_buffer.get(channel, np.empty((0, self.window_size))).shape[
            0
        ]

    def removable_indices_mask(self, channel: str) -> npt.NDArray[np.bool_]:
        """Returns a boolean mask for removable hypotheses in a channel.

        Args:
            channel: Name of the input channel.

        Returns:
            Boolean array indicating removable hypotheses (age >= min_age).
        """
        return self.hyp_age[channel] >= self.min_age

    def add_hyp(self, num_new_hyp: int, channel: str) -> None:
        """Adds new hypotheses to the specified input channel.

        Args:
            num_new_hyp: Number of new hypotheses to add.
            channel: Name of the input channel.
        """
        new_data = np.full((num_new_hyp, self.window_size), np.nan)
        new_age = np.zeros(num_new_hyp, dtype=int)

        if channel not in self.evidence_buffer:
            self.evidence_buffer[channel] = new_data
            self.hyp_age[channel] = new_age
        else:
            self.evidence_buffer[channel] = np.vstack(
                (self.evidence_buffer[channel], new_data)
            )
            self.hyp_age[channel] = np.concatenate((self.hyp_age[channel], new_age))

    def hyp_ages(self, channel: str) -> npt.NDArray[np.int_]:
        """Returns the ages of hypotheses in a channel.

        Args:
            channel: Name of the input channel.
        """
        return self.hyp_age[channel]

    def update(self, values: npt.NDArray[np.float64], channel: str) -> None:
        """Updates all hypotheses in a channel with new evidence values.

        Args:
            values: List or array of new evidence values.
            channel: Name of the input channel.

        Raises:
            ValueError: If the channel doesn't exist or the number of values is
                incorrect.
        """
        if channel not in self.evidence_buffer:
            raise ValueError(f"Channel '{channel}' does not exist.")

        if values.shape[0] != self.total_size(channel):
            raise ValueError(
                f"Expected {self.total_size(channel)} values, but got {len(values)}"
            )

        # Shift evidence buffer by one step
        self.evidence_buffer[channel][:, :-1] = self.evidence_buffer[channel][:, 1:]

        # Add new evidence data
        self.evidence_buffer[channel][:, -1] = values

        # Increment age
        self.hyp_age[channel] += 1

    def calculate_slopes(self, channel: str) -> npt.NDArray[np.float64]:
        """Computes the average slope of hypotheses in a channel.

        This method calculates the slope of the evidence signal for each hypothesis by
        subtracting adjacent values along the time dimension (i.e., computing deltas
        between consecutive evidence values). It then computes the average of these
        differences while ignoring any missing (NaN) values. For hypotheses with no
        valid evidence differences (e.g., all NaNs), the slope is returned as NaN.

        Args:
            channel: Name of the input channel.

        Returns:
            Array of average slopes, one per hypothesis.
        """
        # Calculate the evidence differences
        diffs = np.diff(self.evidence_buffer[channel], axis=1)

        # Count the number of non-NaN values
        valid_steps = np.sum(~np.isnan(diffs), axis=1).astype(np.float64)

        # Set valid steps to Nan to avoid dividing by zero
        valid_steps[valid_steps == 0] = np.nan

        # Return the average slope for each tracked hypothesis, ignoring Nan
        return np.nansum(diffs, axis=1) / valid_steps

    def remove_hyp(self, hyp_ids: npt.NDArray[np.int_], channel: str) -> None:
        """Removes specific hypotheses by index in the specified channel.

        Args:
            hyp_ids: List of hypothesis indices to remove.
            channel: Name of the input channel.
        """
        mask = np.ones(self.total_size(channel), dtype=bool)
        mask[hyp_ids] = False
        self.evidence_buffer[channel] = self.evidence_buffer[channel][mask]
        self.hyp_age[channel] = self.hyp_age[channel][mask]

    def clear_hyp(self, channel: str) -> None:
        """Clears the hypotheses in a specific channel.

        Args:
            channel: Name of the input channel.
        """
        if channel in self.evidence_buffer:
            self.remove_hyp(np.arange(self.total_size(channel)), channel)

    def select_hypotheses(
        self, slope_threshold: float, min_maintained_hyps: int, channel: str
    ) -> HypothesesSelection:
        """Returns a hypotheses selection given a slope threshold.

        A hypothesis is maintained if:
          - Its slope is >= the threshold, OR
          - It is not yet removable due to age.

        Args:
            slope_threshold: Minimum slope value to keep a removable (sufficiently old)
                hypothesis.
            min_maintained_hyps: Minimum number of hypotheses to maintain.
            channel: Name of the input channel.

        Note that the parameter `min_maintained_hyps` overrides the `slope_threshold`
        and the removable mask. These hypotheses will be maintained in spite of their
        slopes and ages.

        Returns:
            A selection of hypotheses to maintain.

        Raises:
            ValueError: If the channel does not exist.
        """
        if channel not in self.evidence_buffer:
            raise ValueError(f"Channel '{channel}' does not exist.")

        slopes = self.calculate_slopes(channel)
        removable_mask = self.removable_indices_mask(channel)

        maintain_mask = (slopes >= slope_threshold) | (~removable_mask)

        # Ensure at least `min_maintained_hyps` are maintained.
        # The needed hypotheses are chosen based on their slopes (i.e. high slopes
        # first)
        num_maintained_hyps = int(maintain_mask.sum())
        num_needed_hyps = max(
            0,
            min(min_maintained_hyps, self.total_size(channel)) - num_maintained_hyps,
        )
        if num_needed_hyps > 0:
            cand_idx = np.where(~maintain_mask)[0]

            # Use all available hyps. No sorting here.
            if cand_idx.size == num_needed_hyps:
                maintain_mask[cand_idx] = True

            # Use hyps with the highest slopes.
            else:
                cand_scores = np.nan_to_num(slopes[cand_idx], nan=-np.inf)
                topk_ix = np.argpartition(cand_scores, num_needed_hyps)[
                    -num_needed_hyps:
                ]
                maintain_mask[cand_idx[topk_ix]] = True

        return HypothesesSelection(maintain_mask)


class HypothesesSelection:
    """Encapsulates the selection of hypotheses to maintain or remove.

    This class stores a boolean mask indicating which hypotheses should be maintained.
    From this mask, it can return the indices and masks for both the maintained and
    removed hypotheses. It also provides convenience constructors for creating a
    selection from maintain/remove masks or from maintain/remove index lists.

    Attributes:
        _maintain_mask: Boolean mask of shape (N,) where True indicates a maintain
            hypothesis and False indicates a remove hypothesis.
    """

    def __init__(self, maintain_mask: npt.NDArray[np.bool_]) -> None:
        """Initializes a HypothesesSelection from a maintain mask.

        Args:
            maintain_mask: Boolean array-like of shape (N,) where True indicates a
                maintained hypothesis and False indicates a removed hypothesis.
        """
        self._maintain_mask = np.asarray(maintain_mask, dtype=bool)

    @classmethod
    def from_maintain_mask(cls, mask: npt.NDArray[np.bool_]) -> HypothesesSelection:
        """Creates a selection from a maintain mask.

        Args:
            mask: Boolean array-like where True indicates a maintained hypothesis.

        Returns:
            A HypothesesSelection instance.

        Note:
            This method is added from completeness, but it is redundant as it calls the
            default class `__init__` function.
        """
        return cls(mask)

    @classmethod
    def from_remove_mask(cls, mask: npt.NDArray[np.bool_]) -> HypothesesSelection:
        """Creates a hypotheses selection from a remove mask.

        Args:
            mask: Boolean array-like where True indicates a hypothesis to remove.

        Returns:
            A HypothesesSelection instance.
        """
        return cls(~mask)

    @classmethod
    def from_maintain_ids(
        cls, total_size: int, ids: npt.NDArray[np.int_]
    ) -> HypothesesSelection:
        """Creates a hypotheses selection from maintain indices.

        Args:
            total_size: Total number of hypotheses.
            ids: Indices of hypotheses to maintain.

        Returns:
            A HypothesesSelection instance.

        Raises:
            IndexError: If any index is out of range [0, total_size).
        """
        mask = np.zeros(int(total_size), dtype=bool)

        if ids.size:
            if ids.min() < 0 or ids.max() >= total_size:
                raise IndexError(f"maintain_ids outside [0, {total_size})")
            mask[np.unique(ids)] = True

        return cls(mask)

    @classmethod
    def from_remove_ids(
        cls, total_size: int, ids: npt.NDArray[np.int_]
    ) -> HypothesesSelection:
        """Creates a selection from remove indices.

        Args:
            total_size: Total number of hypotheses.
            ids: Indices of hypotheses to remove.

        Returns:
            A HypothesesSelection instance.

        Raises:
            IndexError: If any index is out of range [0, total_size).
        """
        mask = np.ones(int(total_size), dtype=bool)

        if ids.size:
            if ids.min() < 0 or ids.max() >= total_size:
                raise IndexError(f"remove_ids outside [0, {total_size})")
            mask[np.unique(ids)] = False

        return cls(mask)

    @property
    def maintain_mask(self) -> npt.NDArray[np.bool_]:
        """Returns the maintain mask."""
        return self._maintain_mask

    @property
    def remove_mask(self) -> npt.NDArray[np.bool_]:
        """Returns the remove mask."""
        return ~self._maintain_mask

    @property
    def maintain_ids(self) -> npt.NDArray[np.int_]:
        """Returns the indices of maintained hypotheses."""
        return np.flatnonzero(self._maintain_mask).astype(int)

    @property
    def remove_ids(self) -> npt.NDArray[np.int_]:
        """Returns the indices of removed hypotheses."""
        return np.flatnonzero(~self._maintain_mask).astype(int)

    def __len__(self) -> int:
        """Returns the total number of hypotheses in the selection."""
        return int(self._maintain_mask.size)


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

    Note:
        The logic of `evidence_threshold_config="all"` can be optimized by
        bypassing the `np.min` function here and bypassing the indexing of
        `np.where` function in the displacer. We want to update all the existing
        hypotheses, therefore there is no need to find the specific indices for
        them in the hypotheses space.

    Raises:
        InvalidEvidenceThresholdConfig: If `evidence_threshold_config` is
            not in the allowed values
    """
    # Return 0 for the threshold if there are no evidence scores
    if evidence_all_channels.size == 0:
        return 0.0

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
            "[int, float, '[int]%', 'mean', "
            "'median', 'all', 'x_percent_threshold']"
        )


@dataclass
class ConsistentHypothesesIds:
    """Contains hypotheses ids for symmetry detection.

    These ids will be updated when using the `ResamplingHypothesesUpdater`.
    The update makes sure the ids are consistent across matching steps despite
    resizing of hypothesis spaces.
    """

    hypotheses_ids: npt.NDArray[np.int_]
    channel_sizes: OrderedDictType[str, int]
    graph_id: str


class InvalidEvidenceThresholdConfig(ValueError):
    """Raised when the evidence update threshold is invalid."""

    pass
