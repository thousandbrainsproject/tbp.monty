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

    def test_repr(self):
        """Test string representation of the ChannelMapper."""
        expected_repr = "ChannelMapper({'A': (0, 5), 'B': (5, 15), 'C': (15, 30)})"
        self.assertEqual(repr(self.mapper), expected_repr)


if __name__ == "__main__":
    unittest.main()
