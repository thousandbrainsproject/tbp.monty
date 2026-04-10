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
        _window_size: Number of past values to consider for slope calculation.
        _min_age: Minimum number of updates before a hypothesis can be considered for
            removal.
        _evidence_buffer: Hypothesis evidence buffer of shape (N, window_size).
        _hyp_age: Hypothesis age counters of shape (N,).
    """

    def __init__(self, window_size: int = 10, min_age: int = 5) -> None:
        """Initializes the EvidenceSlopeTracker.

        Args:
            window_size: Number of evidence points per hypothesis.
            min_age: Minimum number of updates before removal is allowed.
        """
        self._window_size = window_size
        self._min_age = min_age
        self._evidence_buffer: npt.NDArray[np.float64] = np.empty(
            (0, window_size), dtype=np.float64
        )
        self._hyp_age: npt.NDArray[np.int_] = np.empty(0, dtype=int)

    def total_size(self) -> int:
        """Returns the number of tracked hypotheses.

        Returns:
            Number of hypotheses currently tracked.
        """
        return self._evidence_buffer.shape[0]

    def removable_indices_mask(self) -> npt.NDArray[np.bool_]:
        """Returns a boolean mask for removable hypotheses.

        Returns:
            Boolean array indicating removable hypotheses (age >= min_age).
        """
        return self._hyp_age >= self._min_age

    def add_hyp(self, num_new_hyp: int) -> None:
        """Adds new hypotheses.

        Args:
            num_new_hyp: Number of new hypotheses to add.
        """
        new_data = np.full((num_new_hyp, self._window_size), np.nan)
        new_age = np.zeros(num_new_hyp, dtype=int)

        self._evidence_buffer = np.vstack((self._evidence_buffer, new_data))
        self._hyp_age = np.concatenate((self._hyp_age, new_age))

    def hyp_ages(self) -> npt.NDArray[np.int_]:
        """Returns the ages of all hypotheses."""
        return self._hyp_age

    def update(self, values: npt.NDArray[np.float64]) -> None:
        """Updates all hypotheses with new evidence values.

        Args:
            values: Array of new evidence values.

        Raises:
            ValueError: If the number of values doesn't match the number of hypotheses.
        """
        if values.shape[0] != self.total_size():
            raise ValueError(
                f"Expected {self.total_size()} values, but got {len(values)}"
            )

        # Shift evidence buffer by one step
        self._evidence_buffer[:, :-1] = self._evidence_buffer[:, 1:]

        # Add new evidence data
        self._evidence_buffer[:, -1] = values

        # Increment age
        self._hyp_age += 1

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
        diffs = np.diff(self._evidence_buffer, axis=1)

        # Count the number of non-NaN values
        valid_steps = np.sum(~np.isnan(diffs), axis=1).astype(np.float64)

        # Set valid steps to Nan to avoid dividing by zero
        valid_steps[valid_steps == 0] = np.nan

        # Return the average slope for each tracked hypothesis, ignoring Nan
        return np.nansum(diffs, axis=1) / valid_steps

    def remove_hyp(self, hyp_idxs: npt.NDArray[np.int_]) -> None:
        """Removes specific hypotheses by index.

        Args:
            hyp_idxs: Array of hypothesis indices to remove.
        """
        mask = np.ones(self.total_size(), dtype=bool)
        mask[hyp_idxs] = False
        self._evidence_buffer = self._evidence_buffer[mask]
        self._hyp_age = self._hyp_age[mask]

    def clear_hyp(self) -> None:
        """Clears all hypotheses."""
        self._evidence_buffer = np.empty((0, self._window_size), dtype=np.float64)
        self._hyp_age = np.empty(0, dtype=int)

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
        """
        slopes = self.calculate_slopes()
        removable_mask = self.removable_indices_mask()

        mask_to_retain = (slopes >= slope_threshold) | (~removable_mask)

        return HypothesesSelection(mask_to_retain)


class HypothesesSelection:
    """Encapsulates the selection of hypotheses to retain or remove.

    This class stores a boolean mask indicating which hypotheses should be retained.
    From this mask, it can return the indices and masks for both the retained and
    removed hypotheses. It also provides convenience constructors for creating a
    selection from retain/remove masks or from retain/remove index lists.

    Attributes:
        _mask_to_retain: Boolean mask of shape (N,) where True indicates a retained
            hypothesis and False indicates a removed hypothesis.
    """

    def __init__(self, mask_to_retain: npt.NDArray[np.bool_]) -> None:
        """Initializes a HypothesesSelection from a retain mask.

        Args:
            mask_to_retain: Boolean array-like of shape (N,) where True indicates a
                retained hypothesis and False indicates a removed hypothesis.
        """
        self._mask_to_retain = np.asarray(mask_to_retain, dtype=bool)

    @classmethod
    def from_mask_to_retain(cls, mask: npt.NDArray[np.bool_]) -> HypothesesSelection:
        """Creates a selection from a retain mask.

        Args:
            mask: Boolean array-like where True indicates a retained hypothesis.

        Returns:
            A HypothesesSelection instance.

        Note:
            This method is added from completeness, but it is redundant as it calls the
            default class `__init__` function.
        """
        return cls(mask)

    @classmethod
    def from_mask_to_remove(cls, mask: npt.NDArray[np.bool_]) -> HypothesesSelection:
        """Creates a hypotheses selection from a remove mask.

        Args:
            mask: Boolean array-like where True indicates a hypothesis to remove.

        Returns:
            A HypothesesSelection instance.
        """
        return cls(~mask)

    @classmethod
    def from_ids_to_retain(
        cls, total_size: int, ids: npt.NDArray[np.int_]
    ) -> HypothesesSelection:
        """Creates a hypotheses selection from retain indices.

        Args:
            total_size: Total number of hypotheses.
            ids: Indices of hypotheses to retain.

        Returns:
            A HypothesesSelection instance.

        Raises:
            IndexError: If any index is out of range [0, total_size).
        """
        mask = np.zeros(int(total_size), dtype=bool)

        if ids.size:
            if ids.min() < 0 or ids.max() >= total_size:
                raise IndexError(f"ids_to_retain outside [0, {total_size})")
            mask[np.unique(ids)] = True

        return cls(mask)

    @classmethod
    def from_ids_to_remove(
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
                raise IndexError(f"ids_to_remove outside [0, {total_size})")
            mask[np.unique(ids)] = False

        return cls(mask)

    @property
    def mask_to_retain(self) -> npt.NDArray[np.bool_]:
        """Returns the retain mask."""
        return self._mask_to_retain

    @property
    def mask_to_remove(self) -> npt.NDArray[np.bool_]:
        """Returns the remove mask."""
        return ~self._mask_to_retain

    @property
    def ids_to_retain(self) -> npt.NDArray[np.int_]:
        """Returns the indices of retained hypotheses."""
        return np.flatnonzero(self._mask_to_retain).astype(int)

    @property
    def ids_to_remove(self) -> npt.NDArray[np.int_]:
        """Returns the indices of removed hypotheses."""
        return np.flatnonzero(~self._mask_to_retain).astype(int)

    def __len__(self) -> int:
        """Returns the total number of hypotheses in the selection."""
        return self._mask_to_retain.size
