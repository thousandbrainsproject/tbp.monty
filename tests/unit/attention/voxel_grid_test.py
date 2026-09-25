# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import re
import unittest
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import (
    VOXEL_LEVELS,
    Voxel,
    VoxelGrid,
    encode_voxel_grid,
    validate_dataframe,
    voxelize_and_bin_points,
    voxelize_points,
)
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tests.strategies.arrays import (
    float_array_n_by_3,
    float_array_not_1d,
    float_array_not_n_by_3,
)
from tests.unit.attention import strategies


@dataclass
class VoxelizedAndBinnedPoints:
    voxel_size: float
    points: npt.NDArray[np.floating]
    point_ind_to_voxel: list[Voxel]
    voxel_to_point_inds: dict[Voxel, list[int]]
    weights: npt.NDArray[np.floating]


@st.composite
def voxelized_and_binned_points(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = strategies.voxel_sizes,
    weights_strategy: Callable[
        [int], st.SearchStrategy[npt.NDArray[np.floating]]
    ] = strategies.default_attention_system_weights,
) -> VoxelizedAndBinnedPoints:
    """Construct a set of points that are known to lie inside specific voxels.

    Strategy overview
      1. Select voxels that will be occupied.
      2. Assign the number of points that will fall into each voxel.
      3. For each voxel, generate the points that fall inside it.
      4. Generate the weights for each point.

    Returns:
       Voxelized and binned points.
    """
    voxel_size = draw(voxel_size_strategy)

    # 1. Select voxels that will be occupied. Since voxel coordinates depend on
    #    voxel sizes, we have to scale min/max voxel coordinates to make sure all
    #    points will fall in a voxel.
    voxels = draw(strategies.unique_voxels(min_voxels=1))

    # 2. Assign the number of points that will fall into each voxel.
    points_per_voxel = draw(
        st.lists(
            st.integers(min_value=1, max_value=strategies.MAX_POINTS_PER_VOXEL),
            min_size=len(voxels),
            max_size=len(voxels),
        )
    )

    # 3. Generate the points inside each voxel and voxel-to-point mapping.
    points: list[npt.NDArray[np.floating]] = []
    point_ind_to_voxel: list[Voxel] = []
    voxel_to_point_inds: dict[Voxel, list[int]] = defaultdict(list)
    for voxel_ind, voxel in enumerate(voxels):
        for _ in range(points_per_voxel[voxel_ind]):
            voxel_offsets = draw(
                arrays(
                    dtype=np.float64,
                    shape=(3,),
                    elements=st.floats(
                        min_value=strategies.VOXEL_EDGE_TOLERANCE,
                        max_value=1 - strategies.VOXEL_EDGE_TOLERANCE,
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
    weights = draw(weights_strategy(len(points)))

    return VoxelizedAndBinnedPoints(
        voxel_size=voxel_size,
        points=np.stack(points),
        point_ind_to_voxel=point_ind_to_voxel,
        voxel_to_point_inds=voxel_to_point_inds,
        weights=weights,
    )


class VoxelizePointsTest(unittest.TestCase):
    @given(points=float_array_not_n_by_3())
    def test_points_not_n_by_3_raises_value_error(
        self, points: npt.NDArray[np.float64]
    ) -> None:
        with self.assertRaises(ValueError):
            voxelize_points(voxel_size=1.0, points=points)

    @given(binned=voxelized_and_binned_points())
    def test_returned_tuple_list_contains_each_points_voxel(
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        result = voxelize_points(binned.voxel_size, binned.points)
        expected = binned.point_ind_to_voxel
        nptest.assert_array_equal(result, expected)


class VoxelizeAndBinPointsTest(unittest.TestCase):
    @given(points=float_array_n_by_3(), weights=float_array_not_1d())
    def test_weights_not_1d_raises_value_error(
        self, points: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
    ) -> None:
        with self.assertRaises(ValueError):
            voxelize_and_bin_points(
                voxel_size=1.0,
                points=points,
                weights=weights,
            )

    @given(binned=voxelized_and_binned_points())
    def test_returned_dataframe_rows_contain_each_points_voxel_from_voxelize_points_and_weight(  # noqa: E501
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        with patch(
            "tbp.monty.attention.voxel_grid.voxelize_points"
        ) as voxelize_points_mock:
            voxelize_points_mock.return_value = sentinel.voxels
            result = voxelize_and_bin_points(
                voxel_size=binned.voxel_size,
                points=binned.points,
                weights=binned.weights,
            )

        voxels = list(result["voxel"])
        nptest.assert_array_equal(voxels, sentinel.voxels)

        weights = result["weight"]
        nptest.assert_array_equal(weights, binned.weights)

        voxelize_points_mock.assert_called_once_with(binned.voxel_size, binned.points)


@dataclass
class VoxelGridAndPoints:
    voxel_grid: VoxelGrid
    points: npt.NDArray[np.floating]
    weights: npt.NDArray[np.floating]
    point_in_grid: npt.NDArray[np.bool_]


@st.composite
def voxel_grid_and_points(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = strategies.voxel_sizes,
) -> VoxelGridAndPoints:
    """Construct a set of points that are known to lie inside specific voxels.

    Strategy overview
      1. Select distinct voxels and split them into occupied and unoccupied voxels.
         Create a VoxelGrid using the occupied voxels and their (drawn) weights.
      2. Construct points that fall within the grid and their weights. A point's weight
         is equal to its enclosing voxel's weight.
      3. Construct points that do not fall within the grid by generating points inside
         the unoccupied voxels.

    Returns:
       Voxel grid and points.
    """
    voxel_size = draw(voxel_size_strategy)

    # 1. Select distinct voxels and split them into occupied and unoccupied voxels.
    #    Drawing both from one unique list guarantees they don't overlap. Select a
    #    weight for each occupied voxel.
    occupied_and_unoccupied_voxels = draw(strategies.unique_voxels(min_voxels=2))
    num_occupied_voxels = draw(
        st.integers(min_value=1, max_value=len(occupied_and_unoccupied_voxels) - 1)
    )
    occupied_voxels = occupied_and_unoccupied_voxels[:num_occupied_voxels]
    unoccupied_voxels = occupied_and_unoccupied_voxels[num_occupied_voxels:]
    occupied_voxel_weights = draw(
        strategies.default_attention_system_weights(length=len(occupied_voxels))
    )
    voxel_grid = VoxelGrid(
        voxel_size=voxel_size,
        voxels=occupied_voxels,
        weights=occupied_voxel_weights,
    )

    # 2. Construct points that fall within the grid and their weights. A point's weight
    # is equal to its enclosing voxel's weight.
    points_per_voxel = draw(
        st.lists(
            st.integers(min_value=1, max_value=strategies.MAX_POINTS_PER_VOXEL),
            min_size=len(occupied_voxels),
            max_size=len(occupied_voxels),
        )
    )
    points: list[npt.NDArray[np.floating]] = []
    weights: list[float] = []
    point_in_grid: list[bool] = []
    for voxel_ind, voxel in enumerate(occupied_voxels):
        for _ in range(points_per_voxel[voxel_ind]):
            voxel_offsets = draw(
                arrays(
                    dtype=np.float64,
                    shape=(3,),
                    elements=st.floats(
                        min_value=strategies.VOXEL_EDGE_TOLERANCE,
                        max_value=1 - strategies.VOXEL_EDGE_TOLERANCE,
                        exclude_max=True,
                    ),
                )
            )
            points.append((np.array(voxel, dtype=float) + voxel_offsets) * voxel_size)
            weights.append(occupied_voxel_weights[voxel_ind])
            point_in_grid.append(True)

    # 3. Construct points that do not fall within the grid by generating points inside
    #    the unoccupied voxels.
    for voxel in unoccupied_voxels:
        voxel_offsets = draw(
            arrays(
                dtype=np.float64,
                shape=(3,),
                elements=st.floats(
                    min_value=strategies.VOXEL_EDGE_TOLERANCE,
                    max_value=1 - strategies.VOXEL_EDGE_TOLERANCE,
                    exclude_max=True,
                ),
            )
        )
        points.append((np.array(voxel, dtype=float) + voxel_offsets) * voxel_size)
        weights.append(0.0)
        point_in_grid.append(False)

    return VoxelGridAndPoints(
        voxel_grid=voxel_grid,
        points=np.stack(points),
        weights=np.array(weights),
        point_in_grid=np.array(point_in_grid, dtype=bool),
    )


@st.composite
def non_unique_voxels(draw: st.DrawFn) -> list[Voxel]:
    """Draw a list of voxels in which at least one voxel appears more than once.

    Returns:
        Voxels with duplicates.
    """
    coords = st.integers(min_value=-100, max_value=100)
    voxels = draw(st.lists(st.tuples(coords, coords, coords), min_size=1))
    duplicate = draw(st.sampled_from(voxels))
    return draw(st.permutations([*voxels, duplicate]))


class VoxelGridTest(unittest.TestCase):
    def test_from_pandas_validates_input_using_validate_dataframe(self) -> None:
        with patch(
            "tbp.monty.attention.voxel_grid.validate_dataframe"
        ) as validate_dataframe_mock:
            VoxelGrid.from_pandas(voxel_size=MagicMock(), data=sentinel.data)
            validate_dataframe_mock.assert_called_once_with(sentinel.data)

    @given(voxel_grid=strategies.default_voxel_grid())
    def test_init_validates_input_using_validate_dataframe(
        self, voxel_grid: VoxelGrid
    ) -> None:
        with patch(
            "tbp.monty.attention.voxel_grid.validate_dataframe"
        ) as validate_dataframe_mock:
            df = voxel_grid.to_pandas()
            weights = df["weight"].to_numpy()
            voxels = list(df.index)

            VoxelGrid(voxel_size=MagicMock(), voxels=voxels, weights=weights)

            pd.testing.assert_frame_equal(
                validate_dataframe_mock.call_args_list[0][0][0], df
            )

    @given(voxel_grid_and_points=voxel_grid_and_points())
    def test_weights_at_points_returns_weights_for_occupied_voxels_and_fill_value_for_unoccupied_voxels(  # noqa: E501
        self,
        voxel_grid_and_points: VoxelGridAndPoints,
    ) -> None:
        voxel_grid = voxel_grid_and_points.voxel_grid
        points = voxel_grid_and_points.points
        weights = voxel_grid_and_points.weights
        point_in_grid = voxel_grid_and_points.point_in_grid

        result = voxel_grid.weights_at_points(points, fill_value=np.nan)

        expected_weights = weights[point_in_grid]
        actual_weights = result[point_in_grid]
        nptest.assert_array_equal(expected_weights, actual_weights)

        expected_fill_values = np.full(int(np.sum(~point_in_grid)), np.nan)
        actual_fill_values = result[~point_in_grid]
        nptest.assert_array_equal(expected_fill_values, actual_fill_values)


class ValidateDataFrameTest(unittest.TestCase):
    @given(voxel_grid=strategies.default_voxel_grid())
    def test_raises_value_error_if_dataframe_does_not_have_a_multiindex_with_level_names_xyz(  # noqa: E501
        self, voxel_grid: VoxelGrid
    ) -> None:
        df = voxel_grid.to_pandas().reset_index(drop=True)
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "DataFrame must have a multi-index with level names ('x', 'y', 'z')."
            ),
        ):
            validate_dataframe(df)

    @given(voxels=non_unique_voxels())
    def test_raises_value_error_if_dataframe_index_is_not_unique(
        self, voxels: list[Voxel]
    ) -> None:
        df = pd.DataFrame(index=pd.MultiIndex.from_tuples(voxels, names=VOXEL_LEVELS))
        with self.assertRaisesRegex(ValueError, "DataFrame index must be unique."):
            validate_dataframe(df)

    @given(voxel_grid=strategies.default_voxel_grid())
    def test_raises_value_error_if_dataframe_does_not_have_a_weight_column(
        self, voxel_grid: VoxelGrid
    ) -> None:
        df = voxel_grid.to_pandas().drop(columns=["weight"])
        with self.assertRaisesRegex(
            ValueError, "DataFrame must have a 'weight' column."
        ):
            validate_dataframe(df)

    @given(
        voxel_grid=strategies.default_voxel_grid(
            voxels_strategy=strategies.unique_voxels(min_voxels=1)
        )
    )
    def test_raises_value_error_if_weight_column_not_1d(
        self, voxel_grid: VoxelGrid
    ) -> None:
        """Validate dataframe weights column is 1D.

        Note:
            It is not at all straightforward to create a wrong dataframe.

            >1D columns are not supported by pandas. You can force pandas
            into a corrupt state and get strange/undefined behavior in certain
            special cases. I don't know if that means we should or should not spend
            much time testing it. But this test shows one possible case where things
            can go haywire.

            If "weight" has shape (N, 1), pandas silently squeezes it into shape (N,).
            We have to add 2 singleton dimensions like above. Note also that attempting
            to add the reshape a zero-length weight column into (-1, 1, 1) and adding
            it to the dataframe will crash. That's why we set `min_voxels=1` for
            voxel grid creation.
        """
        df = voxel_grid.to_pandas()
        weights = df["weight"].to_numpy()
        del df["weight"]
        df["weight"] = weights.reshape(-1, 1, 1)
        with self.assertRaises(ValueError):
            validate_dataframe(df)


class EncodeVoxelGridTest(unittest.TestCase):
    @given(voxel_grid_and_points=voxel_grid_and_points())
    def test_buffer_encoder_encodes_voxel_grid(
        self,
        voxel_grid_and_points: VoxelGridAndPoints,
    ) -> None:
        voxel_grid = voxel_grid_and_points.voxel_grid
        loaded = json.loads(json.dumps(voxel_grid, cls=BufferEncoder))
        encoded = encode_voxel_grid(voxel_grid)

        self.assertTrue(len(loaded) == len(encoded))
        self.assertTrue(loaded["voxel_size"] == encoded["voxel_size"])
        nptest.assert_array_equal(loaded["voxels"], encoded["voxels"])
        nptest.assert_array_equal(loaded["weight"], encoded["weight"])
