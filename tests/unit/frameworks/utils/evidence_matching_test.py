# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.utils.evidence_matching import (
    EvidenceSlopeTracker,
    extract_unified_displacement,
)


class ExtractUnifiedDisplacementTest(unittest.TestCase):
    """Unit tests for the extract_unified_displacement utility."""

    def test_single_channel(self) -> None:
        d = {"SM_0": np.array([1.0, 2.0, 3.0])}
        result = extract_unified_displacement(d)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_equal_displacements(self) -> None:
        d = {
            "SM_0": np.array([1.0, 0.0, 0.0]),
            "SM_1": np.array([1.0, 0.0, 0.0]),
        }
        result = extract_unified_displacement(d)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_mismatched_displacements_raises(self) -> None:
        d = {
            "SM_0": np.array([1.0, 0.0, 0.0]),
            "SM_1": np.array([0.0, 1.0, 0.0]),
        }
        with self.assertRaises(ValueError):
            extract_unified_displacement(d)

    def test_nearly_equal_within_tolerance(self) -> None:
        d = {
            "SM_0": np.array([1.0, 0.0, 0.0]),
            "SM_1": np.array([1.0 + 1e-7, 0.0, 0.0]),
        }
        result = extract_unified_displacement(d)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])


class EvidenceSlopeTrackerTest(unittest.TestCase):
    """Unit tests for the EvidenceSlopeTracker class."""

    def setUp(self) -> None:
        """Set up a new tracker for each test."""
        self.tracker = EvidenceSlopeTracker(window_size=3, min_age=2)

    def test_add_hypotheses_initializes(self) -> None:
        """Test that hypotheses are correctly initialized."""
        self.tracker.add_hyp(2)
        self.assertEqual(self.tracker.total_size(), 2)
        self.assertEqual(self.tracker.evidence_buffer.shape, (2, 3))
        self.assertTrue(np.all(np.isnan(self.tracker.evidence_buffer)))
        self.assertTrue(np.all(self.tracker.hyp_age == 0))

    def test_update_correctly_shifts_and_sets_values(self) -> None:
        """Test that update correctly shifts previous values and adds new ones."""
        self.tracker.add_hyp(2)
        self.tracker.update(np.array([1.0, 2.0]))
        self.tracker.update(np.array([2.0, 3.0]))
        self.tracker.update(np.array([3.0, 4.0]))

        expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(self.tracker.evidence_buffer, expected)
        np.testing.assert_array_equal(self.tracker.hyp_age, [3, 3])

    def test_update_more_than_window_size_slides_correctly(self) -> None:
        """Test that only the most recent values within window_size affect slope."""
        self.tracker.add_hyp(1)

        # Perform 5 updates; window_size is 3 so only the last 3 should be considered
        self.tracker.update(np.array([0.0]))
        self.tracker.update(np.array([3.0]))
        self.tracker.update(np.array([5.0]))
        self.tracker.update(np.array([4.0]))
        self.tracker.update(np.array([3.0]))

        # Final buffer should be [5.0, 4.0, 3.0]
        expected_buffer = np.array([[5.0, 4.0, 3.0]])
        np.testing.assert_array_equal(self.tracker.evidence_buffer, expected_buffer)

        # Slopes: (3.0 - 4.0) + (4.0 - 5.0) = (-1) + (-1) = -2 / 2 = -1.0
        slopes = self.tracker.calculate_slopes()
        self.assertAlmostEqual(slopes[0], -1.0)

    def test_update_raises_on_wrong_length(self) -> None:
        """Test that update raises ValueError if the length doesn't match."""
        self.tracker.add_hyp(2)
        with self.assertRaises(ValueError):
            self.tracker.update(np.array([1.0]))

    def test_remove_hypotheses_removes_correct_indices(self) -> None:
        """Test removing a specific hypothesis by index."""
        self.tracker.add_hyp(3)
        self.tracker.update(np.array([1.0, 2.0, 3.0]))
        self.tracker.remove_hyp(np.array([1]))
        self.assertEqual(self.tracker.total_size(), 2)
        np.testing.assert_array_equal(self.tracker.evidence_buffer[:, -1], [1.0, 3.0])

    def test_clear_hyp_removes_all_hypotheses(self) -> None:
        """Test that clear_hyp completely removes all hypotheses."""
        self.tracker.add_hyp(4)
        self.tracker.update(np.array([1.0, 2.0, 3.0, 4.0]))

        # Confirm hypotheses were added
        self.assertEqual(self.tracker.total_size(), 4)

        # Clear them
        self.tracker.clear_hyp()

        # Confirm the buffer and age arrays are empty
        self.assertEqual(self.tracker.total_size(), 0)
        self.assertEqual(self.tracker.evidence_buffer.shape[0], 0)
        self.assertEqual(self.tracker.hyp_age.shape[0], 0)

    def test_calculate_slopes_correctly(self) -> None:
        """Test slope calculation over the sliding window."""
        self.tracker.add_hyp(1)
        self.tracker.update(np.array([1.0]))
        self.tracker.update(np.array([2.0]))
        self.tracker.update(np.array([3.0]))

        slopes = self.tracker.calculate_slopes()
        expected_slope = ((2.0 - 1.0) + (3.0 - 2.0)) / 2  # = 1.0
        self.assertAlmostEqual(slopes[0], expected_slope)

    def test_removable_indices_mask_matches_min_age(self) -> None:
        """Test that the removable mask reflects min_age cutoff."""
        self.tracker.add_hyp(3)
        self.tracker.hyp_age[:] = [1, 2, 3]
        mask = self.tracker.removable_indices_mask()
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_select_hypotheses_threshold_and_age(self) -> None:
        """Test that select_hypotheses respects slope threshold and min_age."""
        self.tracker.add_hyp(4)

        # slopes are [1, 0, -1, -1]
        self.tracker.update(np.array([1.0, 2.0, 3.0, 3.0]))
        self.tracker.update(np.array([2.0, 2.0, 2.0, 2.0]))
        self.tracker.update(np.array([3.0, 2.0, 1.0, 1.0]))

        # Force ages so only last hyp is too young to remove.
        self.tracker.hyp_age = np.array([3, 3, 3, 1], dtype=int)

        selection = self.tracker.select_hypotheses(slope_threshold=-0.5)

        # 0,1 have higher slopes, 3 is too young
        expected_keep = np.array([0, 1, 3], dtype=int)
        expected_keep_mask = np.array([True, True, False, True], dtype=bool)

        # lower slope than threshold (-1 < -0.5)
        expected_remove = np.array([2], dtype=int)
        expected_remove_mask = np.array([False, False, True, False], dtype=bool)

        np.testing.assert_array_equal(selection.maintain_ids, expected_keep)
        np.testing.assert_array_equal(selection.remove_ids, expected_remove)
        np.testing.assert_array_equal(selection.maintain_mask, expected_keep_mask)
        np.testing.assert_array_equal(selection.remove_mask, expected_remove_mask)


if __name__ == "__main__":
    unittest.main()
