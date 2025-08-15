# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
)


class ChannelMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a default ChannelMapper instance for testing."""
        self.mapper = ChannelMapper({"A": 5, "B": 10, "C": 15})

    def test_initialization(self) -> None:
        """Test initializing ChannelMapper with predefined sizes."""
        self.assertEqual(self.mapper.channels, ["A", "B", "C"])
        self.assertEqual(self.mapper.total_size, 30)

    def test_channel_range(self) -> None:
        """Test retrieving channel ranges and non-existent channels."""
        self.assertEqual(self.mapper.channel_range("A"), (0, 5))
        self.assertEqual(self.mapper.channel_range("B"), (5, 15))
        self.assertEqual(self.mapper.channel_range("C"), (15, 30))
        with self.assertRaises(ValueError):
            self.mapper.channel_range("D")

    def test_channel_size(self) -> None:
        """Test retrieving total hypotheses for a specific channel."""
        self.assertEqual(self.mapper.channel_size("A"), 5)
        self.assertEqual(self.mapper.channel_size("B"), 10)
        self.assertEqual(self.mapper.channel_size("C"), 15)

        with self.assertRaises(ValueError):
            self.mapper.channel_size("D")

    def test_resize_channel_by_positive(self) -> None:
        """Test increasing channel sizes."""
        self.mapper.resize_channel_by("B", 5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 20))
        self.assertEqual(self.mapper.total_size, 35)

    def test_resize_channel_by_negative(self) -> None:
        """Test decreasing channel sizes."""
        self.mapper.resize_channel_by("B", -5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 10))
        self.assertEqual(self.mapper.total_size, 25)

        with self.assertRaises(ValueError):
            self.mapper.resize_channel_by("A", -10)

    def test_resize_channel_to_valid(self) -> None:
        """Test setting a new size for an existing channel."""
        self.mapper.resize_channel_to("A", 8)
        self.assertEqual(self.mapper.channel_range("A"), (0, 8))
        self.assertEqual(self.mapper.channel_range("B"), (8, 18))
        self.assertEqual(self.mapper.channel_range("C"), (18, 33))
        self.assertEqual(self.mapper.total_size, 33)

    def test_resize_channel_to_invalid_channel(self) -> None:
        """Test resizing a non-existent channel."""
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("Z", 5)

    def test_resize_channel_to_invalid_size(self) -> None:
        """Test resizing a channel to a non-positive size."""
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("B", 0)
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("B", -3)

    def test_add_channel(self) -> None:
        """Test adding a new channel."""
        self.mapper.add_channel("D", 8)
        self.assertIn("D", self.mapper.channels)
        self.assertEqual(self.mapper.channel_range("D"), (30, 38))

        with self.assertRaises(ValueError):
            self.mapper.add_channel("A", 3)

    def test_add_channel_at_position(self) -> None:
        """Test inserting a channel at a specific position."""
        self.mapper.add_channel("X", 7, position=1)
        self.assertEqual(self.mapper.channels, ["A", "X", "B", "C"])
        self.assertEqual(self.mapper.channel_range("X"), (5, 12))
        self.assertEqual(self.mapper.channel_range("B"), (12, 22))
        self.assertEqual(self.mapper.channel_range("C"), (22, 37))

        with self.assertRaises(ValueError):
            self.mapper.add_channel("Y", 5, position=10)

    def test_extract_valid_channel(self) -> None:
        """Test extracting the portion of the array for a valid channel.

        Verifies that the returned data corresponds exactly to the expected slice.
        """
        original = np.arange(30).reshape(30, 1)

        # Channel "B" occupies indices 5 to 15
        extracted = self.mapper.extract(original, "B")

        self.assertTrue(np.array_equal(extracted, original[5:15]))
        self.assertEqual(extracted.shape, (10, 1))

    def test_extract_invalid_channel(self) -> None:
        """Test that extracting from a non-existent channel raises an error."""
        original = np.arange(30).reshape(30, 1)

        with self.assertRaises(ValueError):
            self.mapper.extract(original, "Z")

    def test_extract_hypotheses_valid_channel(self) -> None:
        hypotheses = Hypotheses(
            evidence=np.arange(30).reshape(30, 1),
            locations=np.arange(30).reshape(30, 1),
            poses=np.arange(30).reshape(30, 1),
        )

        # Channel "B" occupies indices 5 to 15
        extracted_hypotheses = self.mapper.extract_hypotheses(hypotheses, "B")

        self.assertTrue(
            np.array_equal(extracted_hypotheses.evidence, hypotheses.evidence[5:15])
        )
        self.assertEqual(extracted_hypotheses.evidence.shape, (10, 1))
        self.assertTrue(
            np.array_equal(extracted_hypotheses.locations, hypotheses.locations[5:15])
        )
        self.assertEqual(extracted_hypotheses.locations.shape, (10, 1))
        self.assertTrue(
            np.array_equal(extracted_hypotheses.poses, hypotheses.poses[5:15])
        )
        self.assertEqual(extracted_hypotheses.poses.shape, (10, 1))
        self.assertEqual(extracted_hypotheses.input_channel, "B")

    def test_update_insert_data(self) -> None:
        """Test inserting new data into a specific channel range.

        Verifies that the new data is inserted at the correct position
        and that the surrounding data is preserved.
        """
        original = np.arange(30).reshape(30, 1)
        new_data = np.array([[100], [101], [102]])

        # Channel "B" originally occupies indices 5 to 15
        updated = self.mapper.update(original, "B", new_data)

        # original size changed from 30 to 23 (i.e., 30 - (10-3))
        self.assertEqual(updated.shape[0], 23)

        # The rest of the data stayed the same
        self.assertTrue(np.array_equal(updated[5:8], new_data))
        self.assertTrue(np.array_equal(updated[:5], original[:5]))
        self.assertTrue(np.array_equal(updated[8:], original[15:]))

    def test_update_invalid_channel(self) -> None:
        """Test that updating a non-existent channel raises an error."""
        original = np.zeros((30, 1))
        new_data = np.ones((3, 1))

        with self.assertRaises(ValueError):
            self.mapper.update(original, "Z", new_data)

    def test_repr(self) -> None:
        """Test string representation of the ChannelMapper."""
        expected_repr = "ChannelMapper({'A': (0, 5), 'B': (5, 15), 'C': (15, 30)})"
        self.assertEqual(repr(self.mapper), expected_repr)


class EvidenceSlopeTrackerTest(unittest.TestCase):
    """Unit tests for the EvidenceSlopeTracker class with channel-specific behavior."""

    def setUp(self) -> None:
        """Set up a new tracker and test channel for each test."""
        self.tracker = EvidenceSlopeTracker(window_size=3, min_age=2)
        self.channel = "patch"

    def test_add_hypotheses_initializes_channel(self) -> None:
        """Test that hypotheses are correctly initialized in a new channel."""
        self.tracker.add_hyp(2, self.channel)
        self.assertEqual(self.tracker.total_size(self.channel), 2)
        self.assertEqual(self.tracker.evidence_buffer[self.channel].shape, (2, 3))
        self.assertTrue(np.all(np.isnan(self.tracker.evidence_buffer[self.channel])))
        self.assertTrue(np.all(self.tracker.hyp_age[self.channel] == 0))

    def test_update_correctly_shifts_and_sets_values(self) -> None:
        """Test that update correctly shifts previous values and adds new ones."""
        self.tracker.add_hyp(2, self.channel)
        self.tracker.update(np.array([1.0, 2.0]), self.channel)
        self.tracker.update(np.array([2.0, 3.0]), self.channel)
        self.tracker.update(np.array([3.0, 4.0]), self.channel)

        expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(
            self.tracker.evidence_buffer[self.channel], expected
        )
        np.testing.assert_array_equal(self.tracker.hyp_age[self.channel], [3, 3])

    def test_update_more_than_window_size_slides_correctly(self) -> None:
        """Test that only the most recent values within window_size affect slope."""
        self.tracker.add_hyp(1, self.channel)

        # Perform 5 updates; window_size is 3 so only the last 3 should be considered
        self.tracker.update(np.array([0.0]), self.channel)
        self.tracker.update(np.array([3.0]), self.channel)
        self.tracker.update(np.array([5.0]), self.channel)
        self.tracker.update(np.array([4.0]), self.channel)
        self.tracker.update(np.array([3.0]), self.channel)

        # Final buffer should be [5.0, 4.0, 3.0]
        expected_buffer = np.array([[5.0, 4.0, 3.0]])
        np.testing.assert_array_equal(
            self.tracker.evidence_buffer[self.channel], expected_buffer
        )

        # Slopes: (3.0 - 4.0) + (4.0 - 5.0) = (-1) + (-1) = -2 / 2 = -1.0
        slopes = self.tracker.calculate_slopes(self.channel)
        self.assertAlmostEqual(slopes[0], -1.0)

    def test_update_raises_on_wrong_length(self) -> None:
        """Test that update raises ValueError if the length doesn't match hypotheses."""
        self.tracker.add_hyp(2, self.channel)
        with self.assertRaises(ValueError):
            self.tracker.update(np.array([1.0]), self.channel)

    def test_remove_hypotheses_removes_correct_indices(self) -> None:
        """Test removing a specific hypothesis by index."""
        self.tracker.add_hyp(3, self.channel)
        self.tracker.update(np.array([1.0, 2.0, 3.0]), self.channel)
        self.tracker.remove_hyp(np.array([1]), self.channel)
        self.assertEqual(self.tracker.total_size(self.channel), 2)
        np.testing.assert_array_equal(
            self.tracker.evidence_buffer[self.channel][:, -1], [1.0, 3.0]
        )

    def test_clear_hyp_removes_all_hypotheses(self) -> None:
        """Test that clear_hyp completely removes all hypotheses in a channel."""
        self.tracker.add_hyp(4, self.channel)
        self.tracker.update(np.array([1.0, 2.0, 3.0, 4.0]), self.channel)

        # Confirm hypotheses were added
        self.assertEqual(self.tracker.total_size(self.channel), 4)

        # Clear them
        self.tracker.clear_hyp(self.channel)

        # Confirm the buffer and age arrays are empty
        self.assertEqual(self.tracker.total_size(self.channel), 0)
        self.assertEqual(self.tracker.evidence_buffer[self.channel].shape[0], 0)
        self.assertEqual(self.tracker.hyp_age[self.channel].shape[0], 0)

    def test_calculate_slopes_correctly(self) -> None:
        """Test slope calculation over the sliding window."""
        self.tracker.add_hyp(1, self.channel)
        self.tracker.update(np.array([1.0]), self.channel)
        self.tracker.update(np.array([2.0]), self.channel)
        self.tracker.update(np.array([3.0]), self.channel)

        slopes = self.tracker.calculate_slopes(self.channel)
        expected_slope = ((2.0 - 1.0) + (3.0 - 2.0)) / 2  # = 1.0
        self.assertAlmostEqual(slopes[0], expected_slope)

    def test_removable_indices_mask_matches_min_age(self) -> None:
        """Test that the removable mask reflects min_age cutoff."""
        self.tracker.add_hyp(3, self.channel)
        self.tracker.hyp_age[self.channel][:] = [1, 2, 3]
        mask = self.tracker.removable_indices_mask(self.channel)
        np.testing.assert_array_equal(mask, [False, True, True])

    def test_select_hypotheses_threshold_and_age(self) -> None:
        """Test that select_hypotheses respects slope threshold and min_age."""
        self.tracker.add_hyp(4, self.channel)

        # slopes are [1, 0, -1, -1]
        self.tracker.update(np.array([1.0, 2.0, 3.0, 3.0]), self.channel)
        self.tracker.update(np.array([2.0, 2.0, 2.0, 2.0]), self.channel)
        self.tracker.update(np.array([3.0, 2.0, 1.0, 1.0]), self.channel)

        # Force ages so only last hyp is too young to remove.
        self.tracker.hyp_age[self.channel] = np.array([3, 3, 3, 1], dtype=int)

        selection = self.tracker.select_hypotheses(
            slope_threshold=-0.5, channel=self.channel
        )

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
