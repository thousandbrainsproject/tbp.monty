# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import numpy.typing as npt
from skimage.segmentation import slic

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
)


class SlicMerge(SegmentationStrategy):
    """Scikit-image SLIC superpixel segmentation with region merging."""

    def __init__(
        self,
        n_seeds: int = 50,
        compactness: float = 10.0,
        max_iter: int = 10,
        sigma: float = 1.0,
        enforce_connectivity: bool = True,
        min_size_factor: float = 0.5,
        max_size_factor: float = 3.0,
        segment_in_lab: bool = True,
        merge_threshold: float = 10,
        merge_in_lab: bool = True,
    ) -> None:
        """Initialize the SLIC superpixel segmentation with region merging strategy.

        Args:
            n_seeds: Number of superpixels to generate, though this is approximate.
                SLIC initializes seeds on a reglar lattice, so the actual number of
                superpixels may be larger.
            compactness: Balances color proximity and space proximity. Higher values
                give more weight to space proximity, resulting in more square
                superpixels.
            max_iter: Maximum number of iterations for SLIC.
            sigma: Width of Gaussian smoothing kernel for pre-processing.
            enforce_connectivity: Whether to enforce connectivity of superpixels.
            min_size_factor: Minimum size of superpixels as a fraction of average size.
            max_size_factor: Maximum size of superpixels as a fraction of average size.
            segment_in_lab: Whether to perform segmentation in LAB color space.
            merge_threshold: Color distance threshold for merging superpixels,
                where "color distance" refers to the CIE76. Standard values:
                    < 1: not perceptible
                    1-2: perceptible through close observation
                    2-10: perceptible at a glance
                    10-50: colors are more similar than opposite
                    > 50: colors are more opposite than similar
            merge_in_lab: Whether to compute color distance in LAB color space.
                Default is True, which is recommended for perceptual color distance.
        """
        self._n_seeds = n_seeds
        self._compactness = compactness
        self._max_iter = max_iter
        self._sigma = sigma
        self._enforce_connectivity = enforce_connectivity
        self._min_size_factor = min_size_factor
        self._max_size_factor = max_size_factor
        self._segment_in_lab = segment_in_lab
        self._merge_threshold = merge_threshold
        self._merge_in_lab = merge_in_lab

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        rgba: np.ndarray,
        depth: np.ndarray | None = None,  # noqa: ARG002
        locations: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        # Fixation is at center of image
        image = rgba[:, :, :3]
        h, w = image.shape[:2]
        y, x = h // 2, w // 2

        # Run SLIC to get initial superpixel segmentation. It returns a 2D image
        # where each pixel holds the (integer-valued) ID of the region it was
        # assigned to.
        region_image: npt.NDArray[np.integer] = slic(
            image,
            n_segments=self._n_seeds,
            compactness=self._compactness,
            max_num_iter=self._max_iter,
            sigma=self._sigma,
            spacing=None,  # Use default spacing
            convert2lab=self._segment_in_lab,
            enforce_connectivity=self._enforce_connectivity,
            min_size_factor=self._min_size_factor,
            max_size_factor=self._max_size_factor,
            start_label=0,
            mask=None,
            channel_axis=-1,
        )

        # Get a version of the input image that's in the color space we want to
        # use for merging. By default, this is the LAB color space.
        if self._merge_in_lab:
            merge_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            merge_thresh = self._merge_threshold
        else:
            merge_image = image.astype(np.float32)
            merge_thresh = self._merge_threshold * 2.55  # Scale to 0-255 range

        # Compute mean color per superpixel
        n_regions = region_image.max() + 1
        region_colors = np.zeros((n_regions, 3), dtype=np.float32)

        # Vectorized computation of mean colors
        for lbl in range(n_regions):
            mask = region_image == lbl
            if mask.any():
                region_colors[lbl] = merge_image[mask].mean(axis=0)

        # Build adjacency graph
        adj: list[set[int]] = [set() for _ in range(n_regions)]

        # - Regions are adjacent to their left/right neights.
        h_neighbors = region_image[:, :-1] != region_image[:, 1:]
        for i, j in zip(*np.where(h_neighbors)):
            a, b = region_image[i, j], region_image[i, j + 1]
            adj[a].add(b)
            adj[b].add(a)

        # - Regions are adjacent to their neighbors above and below.
        v_neighbors = region_image[:-1, :] != region_image[1:, :]
        for i, j in zip(*np.where(v_neighbors)):
            a, b = region_image[i, j], region_image[i + 1, j]
            adj[a].add(b)
            adj[b].add(a)

        # Breadth-first merge from the point of fixation.
        central_region = region_image[y, x]
        accepted_regions: set[int] = {
            central_region
        }  # foreground = attentional region.
        visited = {central_region}
        queue = deque([central_region])
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Compute mean color distance between current and neighbor superpixels.
                # If the colors are below the merging threshold, add the new region
                # to the region set.
                color_distance = np.linalg.norm(
                    region_colors[current] - region_colors[neighbor]
                )
                if color_distance < merge_thresh:
                    accepted_regions.add(neighbor)
                    queue.append(neighbor)

        # Create output mask from the set of accepted regions.
        mask = np.zeros((h, w), dtype=np.uint8)
        for lbl in accepted_regions:
            mask[region_image == lbl] = 1

        return mask
