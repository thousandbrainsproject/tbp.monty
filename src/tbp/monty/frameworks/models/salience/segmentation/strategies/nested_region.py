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


class NestedRegion(SegmentationStrategy):
    """Nested part/whole segmentation via barrier compartments + marker watershed.

    Motivating case: an object (e.g. a marble cube) carrying a sticker/logo.
    Fixate the sticker -- get the sticker (including its text, which is a
    different color). Fixate the surface -- get the surface *without* the
    sticker. Neither pure color-merging nor pure edge-compartments can do
    both: color merging leaks through soft shading edges and can't bridge
    text, and edge compartments fragment on dense detail.

    The combination here:

    1. **Barrier map** -- gradient magnitude above ``edge_thresh`` (the
       boundary-strength knob: weak texture/shading edges are ignored, real
       part boundaries survive).
    2. **Compartments** -- connected components of barrier-free space, each
       with a mean LAB color.
    3. **Seed + global color match** -- the fixated compartment's color
       selects all similar compartments (marker 1); dissimilar sizable
       compartments become the opposition (marker 2). Global matching lets a
       shading-split surface (e.g. a cylinder body above/below the sticker)
       rejoin.
    4. **Marker watershed** -- assigns the ambiguous barrier pixels (text
       strokes, sticker outline, veins) to whichever region they are
       geometrically inside. This is the asymmetry that makes text join the
       sticker while the sticker outline splits down its middle.
    5. **Capped hole fill** -- enclosed pockets small relative to the region
       are absorbed (glyph interiors), while a large enclosed part (the whole
       sticker seen from a surface fixation) stays excluded.

    The sensor patch is centred on what it fixates, so the fixation is the
    centre of the frame.
    """

    def __init__(
        self,
        edge_thresh: float = 20.0,
        merge_thresh: float = 20.0,
        blur_sigma: float = 2.0,
        hole_frac: float = 0.15,
        min_comp: int = 10,
        min_region_pct: float = 1.0,
        enforce_connected: bool = True,
    ) -> None:
        """Initialize the nested-region segmentation strategy.

        Args:
            edge_thresh: Gradient magnitude above which a pixel is a barrier.
                Lower means more edges count as real boundaries (finer parts);
                higher means only strong contours survive (larger regions).
            merge_thresh: Max LAB distance from the fixated compartment's mean
                color for another compartment to join the region.
            blur_sigma: Gaussian blur before the gradient. More blur
                suppresses texture edges but widens barrier bands around fine
                detail.
            hole_frac: Enclosed pockets up to this fraction of the region area
                are filled (text glyphs); larger enclosed parts (a whole
                sticker) stay excluded.
            min_comp: Compartments smaller than this get no marker and are
                assigned by the watershed instead.
            min_region_pct: Minimum region size as percent of the image. A
                smaller result ascends to its enclosing parent region (e.g. a
                gap between letters grows to the whole sticker). 0 disables.
            enforce_connected: Keep only the connected component containing
                the fixation. When False, same-colored regions rejoin globally
                (e.g. a surface split by a sticker).
        """
        self._edge_thresh = edge_thresh
        self._merge_thresh = merge_thresh
        self._blur_sigma = blur_sigma
        self._hole_frac = hole_frac
        self._min_comp = min_comp
        self._min_region_pct = min_region_pct
        self._enforce_connected = enforce_connected

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        rgba: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
        locations: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
    ) -> npt.NDArray[np.uint8]:
        """Segment the part or whole containing the centre of the frame.

        Args:
            ctx: The runtime context.
            rgba: The observed frame.
            depth: Unused.
            locations: Unused.

        Returns:
            A binary mask of the same height and width as ``rgba``, non-zero
            on the segmented region.
        """
        image = np.ascontiguousarray(rgba[..., :3])
        height, width = image.shape[:2]
        fix_y, fix_x = height // 2, width // 2

        # 1. barrier map: per-channel Scharr magnitude on a lightly blurred
        # image
        blurred = image.astype(np.float32)
        if self._blur_sigma > 0:
            blurred = cv2.GaussianBlur(blurred, (0, 0), float(self._blur_sigma))
        grad_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        # /16 normalizes the Scharr kernel weight so the threshold is in
        # intensity-per-pixel units
        grad_mag = np.sqrt((grad_x**2 + grad_y**2).sum(axis=2)) / 16.0
        barrier = grad_mag > float(self._edge_thresh)

        # 2. compartments = connected components of barrier-free space
        n_comps, comp_labels = cv2.connectedComponents(
            (~barrier).astype(np.uint8), connectivity=8
        )
        comp_sizes = np.bincount(comp_labels.ravel(), minlength=n_comps)
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        comp_mean_lab = np.zeros((n_comps, 3), np.float32)
        for channel in range(3):
            comp_mean_lab[:, channel] = np.bincount(
                comp_labels.ravel(),
                weights=image_lab[:, :, channel].ravel(),
                minlength=n_comps,
            ) / np.maximum(comp_sizes, 1)

        # 3. seed compartment; if the fixation is on a barrier pixel (e.g. on
        # text) or in a sliver, snap to the dominant compartment nearby
        min_comp_px = int(self._min_comp)
        min_seed_px = max(min_comp_px, 20)
        fix_comp = int(comp_labels[fix_y, fix_x])
        seed_comp = fix_comp
        if seed_comp == 0 or comp_sizes[seed_comp] < min_seed_px:
            for radius in (4, 8, 14, 22):
                window = comp_labels[
                    max(0, fix_y - radius) : fix_y + radius + 1,
                    max(0, fix_x - radius) : fix_x + radius + 1,
                ].ravel()
                candidates = window[window != 0]
                candidates = candidates[comp_sizes[candidates] >= min_seed_px]
                if candidates.size:
                    comp_ids, hit_counts = np.unique(candidates, return_counts=True)
                    # prefer the compartment most present near the fixation,
                    # weighted toward larger compartments
                    seed_comp = int(
                        comp_ids[np.argmax(hit_counts * np.sqrt(comp_sizes[comp_ids]))]
                    )
                    break
            else:
                # no sizable compartment near the fixation: fall back to the
                # fixated compartment itself, or (on pure barrier) to the
                # fixation's watershed basin alone
                seed_comp = fix_comp

        # 4. global color match -> watershed markers
        if seed_comp != 0:
            color_dist = np.linalg.norm(
                comp_mean_lab - comp_mean_lab[seed_comp], axis=1
            )
            in_region = (color_dist < float(self._merge_thresh)) & (
                comp_sizes >= min_comp_px
            )
            in_region[seed_comp] = True
            in_region[0] = False
        else:
            in_region = np.zeros(n_comps, dtype=bool)

        # the returned segment must contain the fixation: if the fixation sits
        # in a different (snapped-away-from) compartment, pull that
        # compartment in regardless of its color
        if fix_comp != 0:
            in_region[fix_comp] = True

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        def grow_region(in_region: np.ndarray) -> np.ndarray:
            """Watershed-assign boundary pixels between region and opposition.

            Returns:
                The region mask after watershed assignment.

            """
            opposition = (~in_region) & (comp_sizes >= min_comp_px)
            opposition[0] = False
            markers = np.zeros((height, width), np.int32)
            markers[in_region[comp_labels]] = 1
            markers[opposition[comp_labels]] = 2
            # seed the fixation pixel itself: watershed never relabels nonzero
            # seeds, so this guarantees the fixation ends up inside the
            # segment even when it lies on a barrier band
            markers[fix_y, fix_x] = 1
            cv2.watershed(image_bgr, markers)
            mask = markers == 1
            # cv2.watershed stamps the 1-px image border -1 unconditionally;
            # repair it from the adjacent interior so the frame edge keeps its
            # region membership
            mask[0, :] = mask[1, :]
            mask[-1, :] = mask[-2, :]
            mask[:, 0] = mask[:, 1]
            mask[:, -1] = mask[:, -2]
            # optionally keep only the connected piece containing the fixation
            if self._enforce_connected:
                _, piece_labels = cv2.connectedComponents(
                    mask.astype(np.uint8), connectivity=8
                )
                mask = piece_labels == piece_labels[fix_y, fix_x]
            return mask

        region_mask = grow_region(in_region)

        # minimum-size ascent: while the region is too small, re-seed from the
        # dominant compartment surrounding it (its enclosing parent) and
        # regrow -- a gap between letters ascends to the sticker, not to an
        # inflated blob
        min_region_px = float(self._min_region_pct) / 100.0 * height * width
        tried = {seed_comp, fix_comp}
        ring_kernel = np.ones((11, 11), np.uint8)
        for attempt in range(5):
            if region_mask.sum() >= min_region_px:
                break
            # widen the search ring each attempt so it can cross thick barrier
            # bands around the too-small region
            ring = cv2.dilate(
                region_mask.astype(np.uint8), ring_kernel, iterations=attempt + 1
            ).astype(bool)
            ring &= ~region_mask
            ring_comps = comp_labels[ring]
            ring_comps = ring_comps[ring_comps != 0]
            ring_comps = ring_comps[~in_region[ring_comps]]
            ring_comps = ring_comps[comp_sizes[ring_comps] >= min_seed_px]
            ring_comps = ring_comps[~np.isin(ring_comps, list(tried))]
            if not ring_comps.size:
                continue  # ring may be all barrier; next attempt searches wider
            comp_ids, ring_counts = np.unique(ring_comps, return_counts=True)
            parent_comp = int(comp_ids[np.argmax(ring_counts)])
            tried.add(parent_comp)
            # admit the parent and everything similar to it, then regrow
            parent_dist = np.linalg.norm(
                comp_mean_lab - comp_mean_lab[parent_comp], axis=1
            )
            in_region |= (parent_dist < float(self._merge_thresh)) & (
                comp_sizes >= min_comp_px
            )
            in_region[parent_comp] = True
            in_region[0] = False
            region_mask = grow_region(in_region)

        # 5. capped hole fill
        max_hole_px = float(self._hole_frac) * region_mask.sum()
        n_holes, hole_labels = cv2.connectedComponents((~region_mask).astype(np.uint8))
        for hole_id in range(1, n_holes):
            hole = hole_labels == hole_id
            touches_border = (
                hole[0, :].any()
                or hole[-1, :].any()
                or hole[:, 0].any()
                or hole[:, -1].any()
            )
            if touches_border:
                continue
            if hole.sum() <= max_hole_px:
                region_mask |= hole

        return region_mask.astype(np.uint8)
