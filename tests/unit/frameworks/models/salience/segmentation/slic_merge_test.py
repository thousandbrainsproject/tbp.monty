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
from dataclasses import dataclass
from unittest.mock import ANY, MagicMock, Mock, call, patch, sentinel

import cv2
import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from tbp.monty.frameworks.models.salience.segmentation.slic_merge import SlicMerge
from tests.equal import ArrayEqual
from tests.strategies.arrays import uint8_array


class SlicMergeCallTest(unittest.TestCase):
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._segment_image"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.extract_region_colors"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.build_adjacency_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._merge_regions"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.create_mask"
    )
    def test_segments_image(
        self,
        create_mask_mock: MagicMock,
        merge_regions_mock: MagicMock,
        build_adjacency_graph_mock: MagicMock,
        extract_region_colors_mock: MagicMock,
        segment_image_mock: MagicMock,
    ) -> None:
        slic_merge = SlicMerge()
        rgb = MagicMock()

        slic_merge(MagicMock(), rgb)

        segment_image_mock.assert_called_once_with(rgb)
        extract_region_colors_mock.assert_called_once_with(
            ANY, segment_image_mock.return_value
        )
        build_adjacency_graph_mock.assert_called_once_with(
            segment_image_mock.return_value, ANY
        )
        merge_regions_mock.assert_called_once_with(
            segment_image_mock.return_value, ANY, ANY
        )
        create_mask_mock.assert_called_once_with(
            ANY, segment_image_mock.return_value, ANY
        )

    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._segment_image"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.extract_region_colors"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.build_adjacency_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._merge_regions"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.create_mask"
    )
    def test_extracts_region_colors(
        self,
        create_mask_mock: MagicMock,  # noqa: ARG002
        merge_regions_mock: MagicMock,
        build_adjacency_graph_mock: MagicMock,
        extract_region_colors_mock: MagicMock,
        segment_image_mock: MagicMock,
    ) -> None:
        slic_merge = SlicMerge()
        rgb = MagicMock()
        n_regions_sentinel = 5
        region_colors_sentinel = np.zeros((n_regions_sentinel, 3))
        extract_region_colors_mock.return_value = region_colors_sentinel

        slic_merge(MagicMock(), rgb)

        extract_region_colors_mock.assert_called_once_with(
            rgb, segment_image_mock.return_value
        )
        build_adjacency_graph_mock.assert_called_once_with(ANY, n_regions_sentinel)
        merge_regions_mock.assert_called_once_with(ANY, region_colors_sentinel, ANY)

    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._segment_image"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.extract_region_colors"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.build_adjacency_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._merge_regions"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.create_mask"
    )
    def test_builds_adjacency_graph(
        self,
        create_mask_mock: MagicMock,  # noqa: ARG002
        merge_regions_mock: MagicMock,
        build_adjacency_graph_mock: MagicMock,
        extract_region_colors_mock: MagicMock,
        segment_image_mock: MagicMock,
    ) -> None:
        slic_merge = SlicMerge()
        rgb = MagicMock()
        n_regions_sentinel = 5
        extract_region_colors_mock.return_value = np.zeros((n_regions_sentinel, 3))

        slic_merge(MagicMock(), rgb)

        build_adjacency_graph_mock.assert_called_once_with(
            segment_image_mock.return_value, n_regions_sentinel
        )
        merge_regions_mock.assert_called_once_with(
            ANY, ANY, build_adjacency_graph_mock.return_value
        )

    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._segment_image"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.extract_region_colors"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.build_adjacency_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._merge_regions"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.create_mask"
    )
    def test_merges_regions(
        self,
        create_mask_mock: MagicMock,
        merge_regions_mock: MagicMock,
        build_adjacency_graph_mock: MagicMock,
        extract_region_colors_mock: MagicMock,
        segment_image_mock: MagicMock,
    ) -> None:
        slic_merge = SlicMerge()
        rgb = MagicMock()
        region_colors_sentinel = np.zeros((5, 3))
        extract_region_colors_mock.return_value = region_colors_sentinel

        slic_merge(MagicMock(), rgb)

        merge_regions_mock.assert_called_once_with(
            segment_image_mock.return_value,
            region_colors_sentinel,
            build_adjacency_graph_mock.return_value,
        )
        create_mask_mock.assert_called_once_with(
            ANY, ANY, merge_regions_mock.return_value
        )

    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._segment_image"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.extract_region_colors"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.build_adjacency_graph"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge._merge_regions"
    )
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.create_mask"
    )
    def test_creates_mask(
        self,
        create_mask_mock: MagicMock,
        merge_regions_mock: MagicMock,
        build_adjacency_graph_mock: MagicMock,  # noqa: ARG002
        extract_region_colors_mock: MagicMock,
        segment_image_mock: MagicMock,
    ) -> None:
        slic_merge = SlicMerge()
        rgb = MagicMock()
        rgb.shape.__getitem__.return_value = sentinel.rgb_shape
        extract_region_colors_mock.return_value = (
            MagicMock(),
            MagicMock(),
        )

        mask = slic_merge(MagicMock(), rgb)

        create_mask_mock.assert_called_once_with(
            sentinel.rgb_shape,
            segment_image_mock.return_value,
            merge_regions_mock.return_value,
        )
        self.assertIs(mask, create_mask_mock.return_value)

class SlicMergeSegmentImageTest(unittest.TestCase):
    @patch("tbp.monty.frameworks.models.salience.segmentation.slic_merge.slic")
    def test_invokes_slic_with_configured_parameters(
        self, slic_mock: MagicMock
    ) -> None:
        slic_merge = SlicMerge()

        result = slic_merge._segment_image(sentinel.rgb)

        slic_mock.assert_called_once_with(
            sentinel.rgb,
            n_segments=slic_merge._n_seeds,
            compactness=slic_merge._compactness,
            max_num_iter=slic_merge._max_iter,
            sigma=slic_merge._sigma,
            spacing=None,
            convert2lab=True,
            enforce_connectivity=True,
            min_size_factor=slic_merge._min_size_factor,
            max_size_factor=slic_merge._max_size_factor,
            start_label=0,
            mask=None,
            channel_axis=-1,
        )
        self.assertIs(result, slic_mock.return_value)

class SlicMergeExtractRegionColorsTest(unittest.TestCase):
    @given(cvt_color_lab=uint8_array(shape=(3,)))
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.compute_superpixel_mean_colors"
    )
    @patch("tbp.monty.frameworks.models.salience.segmentation.slic_merge.cv2.cvtColor")
    def test_converts_to_normalized_lab_image(
        self,
        cvt_color_mock: MagicMock,
        compute_superpixel_mean_colors_mock: MagicMock,
        cvt_color_lab: npt.NDArray[np.uint8],
    ) -> None:
        cvt_color_mock.return_value = cvt_color_lab
        merge_image = cvt_color_lab.astype(np.float32) / 255.0

        SlicMerge.extract_region_colors(sentinel.rgb, MagicMock())

        cvt_color_mock.assert_called_once_with(sentinel.rgb, cv2.COLOR_RGB2LAB)
        compute_superpixel_mean_colors_mock.assert_called_once_with(
            ANY, ArrayEqual(merge_image)
        )

    @given(cvt_color_lab=uint8_array(shape=(3,)))
    @patch(
        "tbp.monty.frameworks.models.salience.segmentation.slic_merge.SlicMerge.compute_superpixel_mean_colors"
    )
    @patch("tbp.monty.frameworks.models.salience.segmentation.slic_merge.cv2.cvtColor")
    def test_computes_superpixel_mean_colors(
        self,
        cvt_color_mock: MagicMock,
        compute_superpixel_mean_colors_mock: MagicMock,
        cvt_color_lab: npt.NDArray[np.uint8],
    ) -> None:
        cvt_color_mock.return_value = cvt_color_lab
        merge_image = cvt_color_lab.astype(np.float32) / 255.0

        result = SlicMerge.extract_region_colors(MagicMock(), sentinel.region_image)

        compute_superpixel_mean_colors_mock.assert_called_once_with(
            sentinel.region_image, ArrayEqual(merge_image)
        )
        self.assertIs(result, compute_superpixel_mean_colors_mock.return_value)
