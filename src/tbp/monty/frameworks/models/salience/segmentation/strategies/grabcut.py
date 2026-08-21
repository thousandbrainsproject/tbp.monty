# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
)


class GrabCut(SegmentationStrategy):
    """Graph-cut foreground/background separation seeded from a fixation disk.

    OpenCV's GrabCut fits per-region color mixture models and cuts the image
    into foreground and background. Seeding: a disk around the fixation is
    probable foreground (its inner third definite foreground), a thin border
    ring is definite background (the fixated object is assumed not to touch
    the image edge), and everything else is probable background — which,
    unlike definite background, lets the cut grow past the seed disk when
    the object is larger than it.

    The sensor patch is centred on what it fixates, so the fixation is the
    centre of the frame.
    """

    def __init__(self, seed_radius: int = 30, iterations: int = 5) -> None:
        """Initialize the GrabCut segmentation strategy.

        Args:
            seed_radius: Radius of the initial foreground disk, in pixels.
                The inner third is definite foreground, the rest probable
                foreground. The region can still grow beyond this radius.
            iterations: Number of GrabCut iterations. More iterations refine
                the segmentation but take longer.
        """
        self._seed_radius = seed_radius
        self._iterations = iterations

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        rgba: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
        locations: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
    ) -> npt.NDArray[np.uint8]:
        """Segment the foreground containing the centre of the frame.

        Args:
            ctx: The runtime context.
            rgba: The observed frame.
            depth: Unused.
            locations: Unused.

        Returns:
            A binary mask of the same height and width as ``rgba``, non-zero
            on the segmented foreground.
        """
        image = np.ascontiguousarray(rgba[..., :3])
        height, width = image.shape[:2]
        fix_y, fix_x = height // 2, width // 2

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(1, min(height, width) // 50)
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        # pixels well inside the seed disk = definite foreground
        inner_radius = max(1, int(self._seed_radius) // 3)
        cv2.circle(gc_mask, (fix_x, fix_y), int(self._seed_radius), cv2.GC_PR_FGD, -1)
        cv2.circle(gc_mask, (fix_x, fix_y), inner_radius, cv2.GC_FGD, -1)

        # grabCut needs at least one background pixel; if the seed disk
        # swallowed the whole image (tiny image / huge radius), give up and
        # call everything foreground
        if not np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)):
            return np.ones((height, width), dtype=np.uint8)

        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image_bgr,
            gc_mask,
            None,
            bg_model,
            fg_model,
            int(self._iterations),
            cv2.GC_INIT_WITH_MASK,
        )

        foreground = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        return foreground.astype(np.uint8)
