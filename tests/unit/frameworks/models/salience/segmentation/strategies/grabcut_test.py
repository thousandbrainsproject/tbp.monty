# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.strategies import GrabCut

# Frame edge length for the synthetic scenes.
SIZE = 64


def scene(object_slice: tuple[slice, slice] | None = None) -> np.ndarray:
    """Build a dark RGBA frame, optionally with a bright centered object.

    Args:
        object_slice: (rows, cols) slices to paint bright red, or None for a
            uniform frame.

    Returns:
        A (SIZE, SIZE, 4) uint8 RGBA image.
    """
    rgba = np.full((SIZE, SIZE, 4), 30, dtype=np.uint8)
    rgba[..., 3] = 255
    if object_slice is not None:
        rgba[object_slice] = (220, 60, 60, 255)
    return rgba


class GrabCutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(42))
        self.strategy = GrabCut(seed_radius=10)

    def test_returns_binary_uint8_mask_of_frame_shape(self) -> None:
        mask = self.strategy(self.ctx, scene((slice(20, 44), slice(20, 44))))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (SIZE, SIZE))
        self.assertTrue(np.isin(mask, (0, 1)).all())

    def test_fixated_object_is_foreground_background_is_not(self) -> None:
        mask = self.strategy(self.ctx, scene((slice(20, 44), slice(20, 44))))
        mask = mask.astype(bool)
        self.assertTrue(mask[SIZE // 2, SIZE // 2])
        # The object interior is in; the corners are out.
        self.assertTrue(mask[24:40, 24:40].all())
        self.assertFalse(mask[:8, :8].any())
        self.assertFalse(mask[-8:, -8:].any())

    def test_region_grows_beyond_the_seed_disk(self) -> None:
        # A 40x40 object dwarfs the radius-10 seed disk; probable-background
        # initialization still lets the cut claim all of it.
        mask = self.strategy(self.ctx, scene((slice(12, 52), slice(12, 52))))
        self.assertTrue(mask.astype(bool)[14:50, 14:50].all())

    def test_seed_disk_swallowing_the_frame_returns_all_foreground(self) -> None:
        strategy = GrabCut(seed_radius=3 * SIZE)
        rgba = scene()[: SIZE // 2, : SIZE // 2]
        mask = strategy(self.ctx, rgba)
        self.assertEqual(mask.shape, (SIZE // 2, SIZE // 2))
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
