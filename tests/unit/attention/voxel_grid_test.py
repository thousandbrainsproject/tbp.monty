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
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import numpy.testing as nptest
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import (
    Voxel,
    as_row_points,
    voxelize_and_bin_points,
    voxelize_points,
)

from .strategies import (
    MAX_POINT_COORDINATE,
    MAX_POINTS,
    MAX_POINTS_PER_VOXEL,
    VOXEL_EDGE_TOLERANCE,
    float_features_values,
    point_coordinates,
    points_1d,
    points_2d,
    voxel_sizes,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@st.composite
def invalid_points(draw: st.DrawFn) -> np.ndarray:
    not_3 = st.one_of(
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=4, max_value=5),
    )
    shape = st.one_of(
        # 1d, but not 3 long
        st.tuples(not_3),
        # 2d, but rows are not 3 wide
        st.tuples(st.integers(min_value=0, max_value=MAX_POINTS), not_3),
        # 3d and up, whatever the axis lengths
        st.lists(st.integers(min_value=0, max_value=3), min_size=3, max_size=5).map(
            tuple
        ),
    )
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=point_coordinates,
        )
    )


class AsRowPointsTest(unittest.TestCase):
    @given(points=st.one_of(points_1d, points_2d))
    def test_returns_an_n_by_3_float_array_with_values_unchanged(
        self,
        points: np.ndarray,
    ) -> None:
        returned = as_row_points(points)
        self.assertIsInstance(returned, np.ndarray)
        self.assertEqual(returned.ndim, 2)
        self.assertEqual(returned.shape[1], 3)
        self.assertTrue(np.issubdtype(returned.dtype, np.floating))
        np.testing.assert_array_equal(returned, points.reshape(-1, 3))

    @given(points=invalid_points())
    def test_raises_value_error_given_input_not_shaped_like_row_points(
        self,
        points: np.ndarray,
    ) -> None:
        with self.assertRaises(ValueError):
            as_row_points(points)


@dataclass
class VoxelizedAndBinnedPoints:
    """Points built inside known voxels, and which voxel each went into.

    Voxel k is the k-th distinct voxel met walking along ``points``, so the
    voxels are already in the order ``voxelize_and_bin_points`` reports them.
    """

    voxel_size: float
    points: np.ndarray
    point_ind_to_voxel: list[Voxel]
    voxel_to_point_inds: dict[Voxel, list[int]]
    features: dict[str, np.ndarray]


@st.composite
def voxelized_and_binned_points(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = voxel_sizes,
    feature_strategies: dict[str, Callable[[int], st.SearchStrategy]] | None = None,
) -> VoxelizedAndBinnedPoints:
    """Construct a set of points that are known to lie inside specific voxels.

    Strategy overview
      1. Select voxels that will be occupied.
      2. Assign the number of points that will fall into each voxel.
      3. For each voxel, generate the points that fall inside it.

    Returns:
        points: The generated points inside the voxels.
    """
    voxel_size = draw(voxel_size_strategy)

    # 1. Select voxels that will be occupied. Since voxel coordinates depend on
    #    voxel sizes, we have to scale min/max voxel coordinates to make sure all
    #    points will fall in a voxel.
    min_voxel_coord = int(-MAX_POINT_COORDINATE / voxel_size)
    max_voxel_coord = int(MAX_POINT_COORDINATE / voxel_size)
    voxel_axis_length = max_voxel_coord - min_voxel_coord + 1

    min_occupied_voxels = 1
    max_occupied_voxels = min(voxel_axis_length**3, MAX_POINTS)
    n_occupied_voxels = draw(
        st.integers(min_value=min_occupied_voxels, max_value=max_occupied_voxels)
    )
    voxel_ind_to_voxel = draw(
        st.lists(
            st.tuples(
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
            ),
            min_size=min_occupied_voxels,
            max_size=n_occupied_voxels,
            unique=True,
        )
    )

    # 2. Assign the number of points that will fall into each voxel.
    points_per_voxel = draw(
        st.lists(
            st.integers(min_value=1, max_value=MAX_POINTS_PER_VOXEL),
            min_size=n_occupied_voxels,
            max_size=n_occupied_voxels,
        )
    )

    # 3. Generate the points inside each voxel and voxel-to-point mapping.
    points = []
    point_ind_to_voxel: list[Voxel] = []
    voxel_to_point_inds: dict[Voxel, list[int]] = defaultdict(list)
    for voxel_ind, voxel in enumerate(voxel_ind_to_voxel):
        for _ in range(points_per_voxel[voxel_ind]):
            voxel_offsets = draw(
                arrays(
                    dtype=np.float64,
                    shape=(3,),
                    elements=st.floats(
                        min_value=VOXEL_EDGE_TOLERANCE,
                        max_value=1 - VOXEL_EDGE_TOLERANCE,
                        exclude_max=True,
                    ),
                )
            )
            point = (np.array(voxel, dtype=float) + voxel_offsets) * voxel_size
            point_ind = len(points)
            points.append(point)
            point_ind_to_voxel.append(voxel)
            voxel_to_point_inds[voxel].append(point_ind)

    # Make sure this construction is correct.
    for point_ind, voxel in enumerate(point_ind_to_voxel):
        assert point_ind in voxel_to_point_inds[voxel]
    for voxel, point_inds in voxel_to_point_inds.items():
        for point_ind in point_inds:
            assert point_ind_to_voxel[point_ind] == voxel

    # 4. One value per point for each requested feature.
    features = {
        name: draw(make_strategy(len(points)))
        for name, make_strategy in (feature_strategies or {}).items()
    }

    return VoxelizedAndBinnedPoints(
        voxel_size=voxel_size,
        points=np.stack(points),
        point_ind_to_voxel=point_ind_to_voxel,
        voxel_to_point_inds=voxel_to_point_inds,
        features=features,
    )


class VoxelizePointsTest(unittest.TestCase):
    @given(binned=voxelized_and_binned_points())
    def test_returns_correct_voxels(
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        """`expected` is correct by construction."""
        result = voxelize_points(binned.points, binned.voxel_size)
        expected = binned.point_ind_to_voxel
        nptest.assert_array_equal(result, expected)

    def test_calls_as_row_points(self) -> None:
        # MagicMock, not Mock: the rows get divided by the voxel size.
        with patch(
            "tbp.monty.attention.voxel_grid.as_row_points", return_value=MagicMock()
        ) as as_row_points_mock:
            voxelize_points(sentinel.points, voxel_size=MagicMock())

        as_row_points_mock.assert_called_once_with(sentinel.points)


class VoxelizeAndBinPointsTest(unittest.TestCase):
    @given(binned=voxelized_and_binned_points())
    def test_returns_dataframe_with_correct_voxel(
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        result = voxelize_and_bin_points(binned.points, binned.voxel_size)
        self.assertIsInstance(result, pd.DataFrame)
        nptest.assert_array_equal(list(result["voxel"]), binned.point_ind_to_voxel)

    def test_calls_voxelize_points(self) -> None:
        with patch(
            "tbp.monty.attention.voxel_grid.voxelize_points",
            return_value=[sentinel.voxel],
        ) as voxelize_points_mock:
            voxelize_and_bin_points(sentinel.points, sentinel.voxel_size)

        voxelize_points_mock.assert_called_once_with(
            sentinel.points, sentinel.voxel_size
        )

    @given(
        binned=voxelized_and_binned_points(
            feature_strategies={
                "a": float_features_values,
                "b": float_features_values,
            }
        )
    )
    def test_returns_dataframe_with_correct_features(
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        result = voxelize_and_bin_points(
            binned.points, binned.voxel_size, features=binned.features
        )
        for feature_name, feature_values in binned.features.items():
            nptest.assert_array_equal(result[feature_name], feature_values)
