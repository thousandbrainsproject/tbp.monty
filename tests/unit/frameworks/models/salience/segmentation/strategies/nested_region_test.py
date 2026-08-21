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
from tbp.monty.frameworks.models.salience.segmentation.strategies import NestedRegion

# Frame edge length for the synthetic scenes.
SIZE = 64


def scene(sticker_slice: tuple[slice, slice] | None = None) -> np.ndarray:
    """Build a gray RGBA frame, optionally with a red sticker patch.

    Args:
        sticker_slice: (rows, cols) slices to paint red, or None for a
            uniform frame.

    Returns:
        A (SIZE, SIZE, 4) uint8 RGBA image.
    """
    rgba = np.full((SIZE, SIZE, 4), 128, dtype=np.uint8)
    rgba[..., 3] = 255
    if sticker_slice is not None:
        rgba[sticker_slice] = (200, 40, 40, 255)
    return rgba


class NestedRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(42))
        self.strategy = NestedRegion()

    def test_returns_binary_uint8_mask_of_frame_shape(self) -> None:
        mask = self.strategy(self.ctx, scene())
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (SIZE, SIZE))
        self.assertTrue(np.isin(mask, (0, 1)).all())

    def test_uniform_frame_segments_as_one_region(self) -> None:
        mask = self.strategy(self.ctx, scene())
        self.assertTrue(mask.all())

    def test_fixating_a_sticker_segments_the_sticker_not_the_surface(self) -> None:
        # A sticker centered on the fixation.
        sticker = (slice(18, 46), slice(18, 46))
        mask = self.strategy(self.ctx, scene(sticker)).astype(bool)
        self.assertTrue(mask[SIZE // 2, SIZE // 2])
        # The sticker interior is in; the surface well away from it is out.
        self.assertTrue(mask[24:40, 24:40].all())
        self.assertFalse(mask[:8].any())
        self.assertFalse(mask[-8:].any())

    def test_fixating_the_surface_excludes_the_sticker(self) -> None:
        # A sticker in the corner, away from the central fixation.
        sticker = (slice(4, 32), slice(4, 32))
        mask = self.strategy(self.ctx, scene(sticker)).astype(bool)
        self.assertTrue(mask[SIZE // 2, SIZE // 2])
        # The surface far from the sticker is in; the sticker interior is out.
        self.assertTrue(mask[44:60, 44:60].all())
        self.assertFalse(mask[10:26, 10:26].any())

    def test_small_glyph_inside_the_sticker_is_absorbed(self) -> None:
        # A sticker with a small dark "glyph" dot inside it, off-fixation.
        sticker = (slice(18, 46), slice(18, 46))
        rgba = scene(sticker)
        rgba[24:28, 24:28] = (20, 20, 200, 255)
        mask = self.strategy(self.ctx, rgba).astype(bool)
        # The glyph joins the sticker region rather than punching a hole.
        self.assertTrue(mask[24:28, 24:28].all())
        self.assertFalse(mask[:8].any())


if __name__ == "__main__":
    unittest.main()
