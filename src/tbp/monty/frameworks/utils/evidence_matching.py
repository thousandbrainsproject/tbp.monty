# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class EvidenceSlopeTracker:
    """Tracks the slopes of evidence streams over a sliding window.

    This tracker supports adding, updating, pruning, and analyzing hypotheses
    in a hypothesis space.

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
        evidence_buffer: Hypothesis evidence buffer of shape (N, window_size).
        hyp_age: Hypothesis age counters of shape (N,).
    """

    def __init__(self, window_size: int = 10, min_age: int = 5) -> None:
        """Initializes the EvidenceSlopeTracker.

        Args:
            window_size: Number of evidence points per hypothesis.
            min_age: Minimum number of updates before removal is allowed.
        """
        self.window_size = window_size
        self.min_age = min_age
        self.evidence_buffer: npt.NDArray[np.float64] | None = None
        self.hyp_age: npt.NDArray[np.int_] | None = None

    def total_size(self) -> int:
        """Returns the number of tracked hypotheses.

        Returns:
            Number of hypotheses currently tracked.
        """
        if self.evidence_buffer is None:
            return 0
        return self.evidence_buffer.shape[0]

    def removable_indices_mask(self) -> npt.NDArray[np.bool_]:
        """Returns a boolean mask for removable hypotheses.

        Returns:
            Boolean array indicating removable hypotheses (age >= min_age).
        """
        return self.hyp_age >= self.min_age

    def add_hyp(self, num_new_hyp: int) -> None:
        """Adds new hypotheses.

        Args:
            num_new_hyp: Number of new hypotheses to add.
        """
        new_data = np.full((num_new_hyp, self.window_size), np.nan)
        new_age = np.zeros(num_new_hyp, dtype=int)

        if self.evidence_buffer is None:
            self.evidence_buffer = new_data
            self.hyp_age = new_age
        else:
            self.evidence_buffer = np.vstack((self.evidence_buffer, new_data))
            self.hyp_age = np.concatenate((self.hyp_age, new_age))

    def hyp_ages(self) -> npt.NDArray[np.int_]:
        """Returns the ages of all hypotheses."""
        return self.hyp_age

    def update(self, values: npt.NDArray[np.float64]) -> None:
        """Updates all hypotheses with new evidence values.

        Args:
            values: Array of new evidence values.

        Raises:
            ValueError: If no hypotheses exist or the number of values is incorrect.
        """
        if self.evidence_buffer is None:
            raise ValueError("No hypotheses exist yet.")

        if values.shape[0] != self.total_size():
            raise ValueError(
                f"Expected {self.total_size()} values, but got {len(values)}"
            )

        # Shift evidence buffer by one step
        self.evidence_buffer[:, :-1] = self.evidence_buffer[:, 1:]

        # Add new evidence data
        self.evidence_buffer[:, -1] = values

        # Increment age
        self.hyp_age += 1

    def calculate_slopes(self) -> npt.NDArray[np.float64]:
        """Computes the average slope of all hypotheses.

        This method calculates the slope of the evidence signal for each hypothesis by
        subtracting adjacent values along the time dimension (i.e., computing deltas
        between consecutive evidence values). It then computes the average of these
        differences while ignoring any missing (NaN) values. For hypotheses with no
        valid evidence differences (e.g., all NaNs), the slope is returned as NaN.

        Returns:
            Array of average slopes, one per hypothesis.
        """
        # Calculate the evidence differences
        diffs = np.diff(self.evidence_buffer, axis=1)

        # Count the number of non-NaN values
        valid_steps = np.sum(~np.isnan(diffs), axis=1).astype(np.float64)

        # Set valid steps to Nan to avoid dividing by zero
        valid_steps[valid_steps == 0] = np.nan

        # Return the average slope for each tracked hypothesis, ignoring Nan
        return np.nansum(diffs, axis=1) / valid_steps

    def remove_hyp(self, hyp_ids: npt.NDArray[np.int_]) -> None:
        """Removes specific hypotheses by index.

        Args:
            hyp_ids: Array of hypothesis indices to remove.
        """
        mask = np.ones(self.total_size(), dtype=bool)
        mask[hyp_ids] = False
        self.evidence_buffer = self.evidence_buffer[mask]
        self.hyp_age = self.hyp_age[mask]

    def clear_hyp(self) -> None:
        """Clears all hypotheses."""
        if self.evidence_buffer is not None:
            self.remove_hyp(np.arange(self.total_size()))

    def select_hypotheses(self, slope_threshold: float) -> HypothesesSelection:
        """Returns a hypotheses selection given a slope threshold.

        A hypothesis is maintained if:
          - Its slope is >= the threshold, OR
          - It is not yet removable due to age.

        Args:
            slope_threshold: Minimum slope value to keep a removable (sufficiently old)
                hypothesis.

        Returns:
            A selection of hypotheses to maintain.

        Raises:
            ValueError: If no hypotheses exist.
        """
        if self.evidence_buffer is None:
            raise ValueError("No hypotheses exist yet.")

        slopes = self.calculate_slopes()
        removable_mask = self.removable_indices_mask()

        maintain_mask = (slopes >= slope_threshold) | (~removable_mask)

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
        return self._maintain_mask.size


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

    if isinstance(evidence_threshold_config, (int, float)):
        return evidence_threshold_config

    if evidence_threshold_config == "mean":
        return np.mean(evidence_all_channels)

    if evidence_threshold_config == "median":
        return np.median(evidence_all_channels)

    if isinstance(
        evidence_threshold_config, str
    ) and evidence_threshold_config.endswith("%"):
        percentage_str = evidence_threshold_config.strip("%")
        percentage = float(percentage_str)
        assert percentage >= 0 and percentage <= 100, (
            "Percentage must be between 0 and 100"
        )
        x_percent_of_max = max_global_evidence * (percentage / 100)
        return max_global_evidence - x_percent_of_max

    if evidence_threshold_config == "x_percent_threshold":
        x_percent_of_max = max_global_evidence / 100 * float(x_percent_threshold)
        return max_global_evidence - x_percent_of_max

    if evidence_threshold_config == "all":
        return np.min(evidence_all_channels)

    raise InvalidEvidenceThresholdConfig(
        "evidence_threshold_config not in "
        "[int, float, '[int]%', 'mean', "
        "'median', 'all', 'x_percent_threshold']"
    )


class InvalidEvidenceThresholdConfig(ValueError):
    """Raised when the evidence update threshold is invalid."""

    pass


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
