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

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from tbp.monty.frameworks.models.salience.segmentation.slic_merge import SlicMerge


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
    ):
        slic_merge = SlicMerge()
        rgb = MagicMock()
        extract_region_colors_mock.return_value = (MagicMock(), MagicMock())

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
    ):
        slic_merge = SlicMerge()
        rgb = MagicMock()
        extract_region_colors_mock.return_value = (
            sentinel.n_regions,
            sentinel.region_colors,
        )

        slic_merge(MagicMock(), rgb)

        extract_region_colors_mock.assert_called_once_with(
            rgb, segment_image_mock.return_value
        )
        build_adjacency_graph_mock.assert_called_once_with(
            ANY, sentinel.n_regions
        )
        merge_regions_mock.assert_called_once_with(
            ANY, sentinel.region_colors, ANY
        )

    def test_builds_adjacency_graph(self):
        pass

    def test_merges_regions(self):
        pass

    def test_creates_mask(self):
        pass
