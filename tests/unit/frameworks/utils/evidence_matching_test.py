# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.monty.frameworks.utils.evidence_matching import ChannelMapper


class ChannelMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a default ChannelMapper instance for testing."""
        self.mapper = ChannelMapper({"A": 5, "B": 10, "C": 15})

    def test_initialization(self):
        """Test initializing ChannelMapper with predefined sizes."""
        self.assertEqual(self.mapper.channels, ["A", "B", "C"])
        self.assertEqual(self.mapper.total_size, 30)

    def test_channel_range(self):
        """Test retrieving channel ranges and non-existent channels."""
        self.assertEqual(self.mapper.channel_range("A"), (0, 5))
        self.assertEqual(self.mapper.channel_range("B"), (5, 15))
        self.assertEqual(self.mapper.channel_range("C"), (15, 30))
        with self.assertRaises(ValueError):
            self.mapper.channel_range("D")

    def test_resize_channel_by_positive(self):
        """Test increasing channel sizes."""
        self.mapper.resize_channel_by("B", 5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 20))
        self.assertEqual(self.mapper.total_size, 35)

    def test_resize_channel_by_negative(self):
        """Test decreasing channel sizes."""
        self.mapper.resize_channel_by("B", -5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 10))
        self.assertEqual(self.mapper.total_size, 25)

        with self.assertRaises(ValueError):
            self.mapper.resize_channel_by("A", -10)

    def test_resize_channel_to_valid(self):
        """Test setting a new size for an existing channel."""
        self.mapper.resize_channel_to("A", 8)
        self.assertEqual(self.mapper.channel_range("A"), (0, 8))
        self.assertEqual(self.mapper.channel_range("B"), (8, 18))
        self.assertEqual(self.mapper.channel_range("C"), (18, 33))
        self.assertEqual(self.mapper.total_size, 33)

    def test_resize_channel_to_invalid_channel(self):
        """Test resizing a non-existent channel."""
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("Z", 5)

    def test_resize_channel_to_invalid_size(self):
        """Test resizing a channel to a non-positive size."""
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("B", 0)
        with self.assertRaises(ValueError):
            self.mapper.resize_channel_to("B", -3)

    def test_add_channel(self):
        """Test adding a new channel."""
        self.mapper.add_channel("D", 8)
        self.assertIn("D", self.mapper.channels)
        self.assertEqual(self.mapper.channel_range("D"), (30, 38))

        with self.assertRaises(ValueError):
            self.mapper.add_channel("A", 3)

    def test_add_channel_at_position(self):
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
        import numpy as np

        original = np.arange(30).reshape(30, 1)

        # Channel "B" occupies indices 5 to 15
        extracted = self.mapper.extract(original, "B")

        self.assertTrue(np.array_equal(extracted, original[5:15]))
        self.assertEqual(extracted.shape, (10, 1))

    def test_extract_invalid_channel(self) -> None:
        """Test that extracting from a non-existent channel raises an error."""
        import numpy as np

        original = np.arange(30).reshape(30, 1)

        with self.assertRaises(ValueError):
            self.mapper.extract(original, "Z")

    def test_update_insert_data(self) -> None:
        """Test inserting new data into a specific channel range.

        Verifies that the new data is inserted at the correct position
        and that the surrounding data is preserved.
        """
        import numpy as np

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
        import numpy as np

        original = np.zeros((30, 1))
        new_data = np.ones((3, 1))

        with self.assertRaises(ValueError):
            self.mapper.update(original, "Z", new_data)

    def test_repr(self):
        """Test string representation of the ChannelMapper."""
        expected_repr = "ChannelMapper({'A': (0, 5), 'B': (5, 15), 'C': (15, 30)})"
        self.assertEqual(repr(self.mapper), expected_repr)


if __name__ == "__main__":
    unittest.main()
