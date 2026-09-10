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

from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder


class AddPatchOutlineToViewFinderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.center = (10, 10)
        self.patch_size = 8
        # A pixel that falls on the top edge of the drawn outline square.
        self.edge_pixel = (self.center[0] - self.patch_size // 2, self.center[1])

    def test_rgb_view_finder_gets_blue_outline(self) -> None:
        """A 3-channel RGB view finder is outlined without a broadcasting error."""
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        marked = add_patch_outline_to_view_finder(image, self.center, self.patch_size)
        np.testing.assert_array_equal(marked[self.edge_pixel], [0, 0, 255])

    def test_rgba_view_finder_outline_unchanged(self) -> None:
        """A 4-channel RGBA view finder keeps the original outline value."""
        image = np.zeros((20, 20, 4), dtype=np.uint8)
        marked = add_patch_outline_to_view_finder(image, self.center, self.patch_size)
        np.testing.assert_array_equal(marked[self.edge_pixel], [0, 0, 255, 0])


if __name__ == "__main__":
    unittest.main()
