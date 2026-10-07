# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from skimage.segmentation import slic

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.strategy import (
    SegmentationStrategy,
)


class SlicMerge(SegmentationStrategy):
    """Scikit-image SLIC superpixel segmentation with region merging."""

    _n_seeds: int
    _compactness: float
    _max_iter: int
    _sigma: float
    _enforce_connectivity: bool
    _min_size_factor: float
    _max_size_factor: float
    _merge_threshold: float

    def __init__(
        self,
        n_seeds: int = 70,
        compactness: float = 10.0,
        max_iter: int = 10,
        sigma: float = 1.0,
        enforce_connectivity: bool = True,
        min_size_factor: float = 0.5,
        max_size_factor: float = 3.0,
        merge_threshold: float = 10,
    ) -> None:
        """Initialize the SLIC superpixel segmentation with region merging strategy.

        Args:
            n_seeds: Number of superpixels to generate, though this is approximate.
                SLIC initializes seeds on a regular lattice, so the actual number of
                superpixels may be larger.
            compactness: Balances color proximity and space proximity. Higher values
                give more weight to space proximity, resulting in more square
                superpixels.
            max_iter: Maximum number of iterations for SLIC.
            sigma: Width of Gaussian smoothing kernel for pre-processing.
            enforce_connectivity: Whether to enforce connectivity of superpixels.
            min_size_factor: Minimum superpixel size, as a fraction of the
                nominal size `image_pixels / n_seeds`. Smaller connected
                fragments are merged into an adjacent superpixel.
            max_size_factor: Maximum superpixel size, as a fraction of the
                nominal size. Larger connected regions are split.
            merge_threshold: Color distance threshold for merging superpixels,
                where "color distance" refers to the CIE76. Standard values:
                    < 1: not perceptible
                    1-2: perceptible through close observation
                    2-10: perceptible at a glance
                    10-50: colors are more similar than opposite
                    > 50: colors are more opposite than similar
        """
        # slic parameters
        self._n_seeds = n_seeds
        self._compactness = compactness
        self._max_iter = max_iter
        self._sigma = sigma
        self._enforce_connectivity = enforce_connectivity
        self._min_size_factor = min_size_factor
        self._max_size_factor = max_size_factor
        # post-slic merging parameters
        self._merge_threshold = merge_threshold

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        rgb: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        region_image = self._segment_image(rgb)
        n_regions, region_colors = self.extract_region_colors(rgb, region_image)
        adj = self.build_adjacency_graph(region_image, n_regions)
        accepted_regions = self._merge_regions(region_image, region_colors, adj)
        return self.create_mask(rgb.shape[:2], region_image, accepted_regions)

    @staticmethod
    def create_mask(mask_shape, region_image, accepted_regions):
        # Create output mask from the set of accepted regions.
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for lbl in accepted_regions:
            mask[region_image == lbl] = 1

        return mask

    def _merge_regions(self, region_image, region_colors, adj):
        # Breadth-first merge from the point of fixation.
        height, width = region_image.shape[:2]
        central_region = region_image[height // 2, width // 2]
        accepted_regions: set[int] = {central_region}
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
                if color_distance < self._merge_threshold:
                    accepted_regions.add(neighbor)
                    queue.append(neighbor)
        return accepted_regions

    @staticmethod
    def build_adjacency_graph(region_image, n_regions):
        # Build adjacency graph. adj[region_id] contains the ids of neighboring regions.
        adj: list[set[int]] = [set() for _ in range(n_regions)]

        # - Find each region's left and right neighbors.
        h_neighbors = region_image[:, :-1] != region_image[:, 1:]
        for i, j in zip(*np.where(h_neighbors)):
            a, b = region_image[i, j], region_image[i, j + 1]
            adj[a].add(b)
            adj[b].add(a)

        # - Find each region's top and bottom neighbors.
        v_neighbors = region_image[:-1, :] != region_image[1:, :]
        for i, j in zip(*np.where(v_neighbors)):
            a, b = region_image[i, j], region_image[i + 1, j]
            adj[a].add(b)
            adj[b].add(a)
        return adj

    @staticmethod
    def extract_region_colors(rgb: npt.NDArray[np.uint8], region_image):
        # Get a version of the input image that's in the color space we want to
        # use for merging. By default, this is the LAB color space.
        merge_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Compute mean color per superpixel
        n_regions = region_image.max() + 1
        region_colors = np.zeros((n_regions, 3), dtype=np.float32)
        for lbl in range(n_regions):
            mask = region_image == lbl
            if mask.any():
                region_colors[lbl] = merge_image[mask].mean(axis=0)
        return n_regions, region_colors

    def _segment_image(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[Any]:
        # Run SLIC to get initial superpixel segmentation. It returns a 2D image
        # where each pixel holds the (integer-valued) ID of the region it was
        # assigned to.
        return slic(
            rgb,
            n_segments=self._n_seeds,
            compactness=self._compactness,
            max_num_iter=self._max_iter,
            sigma=self._sigma,
            spacing=None,
            convert2lab=True,
            enforce_connectivity=self._enforce_connectivity,
            min_size_factor=self._min_size_factor,
            max_size_factor=self._max_size_factor,
            start_label=0,
            mask=None,
            channel_axis=-1,
        )
