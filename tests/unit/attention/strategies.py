# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import Voxel, VoxelGrid
from tbp.monty.cmp import Goal
from tests.strategies.cmp import goals_at

MIN_POINT_COORDINATE = -10
MAX_POINT_COORDINATE = 10
MAX_VOXELS = 100
MAX_POINTS_PER_VOXEL = 10
MIN_VOXEL_SIZE = 0.001
MAX_VOXEL_SIZE = 1.0
MAX_VOXEL_COORDINATE = int(MAX_POINT_COORDINATE / MIN_VOXEL_SIZE)
MIN_VOXEL_COORDINATE = -MAX_VOXEL_COORDINATE

VOXEL_EDGE_TOLERANCE = 1e-6
"""Generated points stay this far (as a fraction of a voxel) from voxel faces,
so float error cannot move them into a neighbouring voxel.
"""
MIN_ATTENTION_WEIGHT: float = -1.0
"""Full inhibition."""
MAX_ATTENTION_WEIGHT: float = 1.0
"""Full excitation."""

voxel_sizes = st.floats(min_value=MIN_VOXEL_SIZE, max_value=MAX_VOXEL_SIZE)


@st.composite
def default_attention_system_weights(
    draw: st.DrawFn,
    length: int,
) -> npt.NDArray[np.floating]:
    return draw(
        arrays(
            dtype=np.float64,
            shape=(length,),
            elements=st.floats(
                min_value=MIN_ATTENTION_WEIGHT,
                max_value=MAX_ATTENTION_WEIGHT,
            ),
            fill=st.just(0.0),
        )
    )


@st.composite
def all_negative_default_attention_system_weights(
    draw: st.DrawFn, length: int
) -> npt.NDArray[np.floating]:
    return draw(
        arrays(
            dtype=np.float64,
            shape=(length,),
            elements=st.floats(
                min_value=MIN_ATTENTION_WEIGHT,
                max_value=0.0,
                exclude_max=True,
            ),
        )
    )


@st.composite
def all_positive_default_attention_system_weights(
    draw: st.DrawFn, length: int
) -> npt.NDArray[np.floating]:
    return draw(
        arrays(
            dtype=np.float64,
            shape=(length,),
            elements=st.floats(
                min_value=0.0,
                max_value=MAX_ATTENTION_WEIGHT,
                exclude_min=True,
            ),
        )
    )


@st.composite
def with_positive_default_attention_system_weights(
    draw: st.DrawFn, length: int
) -> npt.NDArray[np.floating]:
    if length <= 1:
        return draw(
            arrays(
                dtype=np.float64,
                shape=(length,),
                elements=st.floats(
                    min_value=0.0,
                    max_value=MAX_ATTENTION_WEIGHT,
                    exclude_min=True,
                ),
            )
        )
    default = length // 2
    positive = length - default
    default_weights = draw(default_attention_system_weights(default))
    positive_weights = draw(all_positive_default_attention_system_weights(positive))
    return np.concatenate([default_weights, positive_weights])


@st.composite
def with_negative_default_attention_system_weights(
    draw: st.DrawFn, length: int
) -> npt.NDArray[np.floating]:
    if length <= 1:
        return draw(
            arrays(
                dtype=np.float64,
                shape=(length,),
                elements=st.floats(
                    min_value=MIN_ATTENTION_WEIGHT,
                    max_value=0.0,
                    exclude_max=True,
                ),
            )
        )
    default = length // 2
    negative = length - default
    default_weights = draw(default_attention_system_weights(default))
    negative_weights = draw(all_negative_default_attention_system_weights(negative))
    return np.concatenate([default_weights, negative_weights])


@st.composite
def unique_voxels(
    draw: st.DrawFn,
    min_voxels: int = 0,
    min_voxel_coord: int = MIN_VOXEL_COORDINATE,
    max_voxel_coord: int = MAX_VOXEL_COORDINATE,
) -> list[Voxel]:
    """Return a list of unique voxels."""
    voxel_axis_length = max_voxel_coord - min_voxel_coord + 1

    min_total_voxels = min_voxels
    max_total_voxels = min(voxel_axis_length**3, MAX_VOXELS)
    return draw(
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
            min_size=min_total_voxels,
            max_size=max_total_voxels,
            unique=True,
        )
    )


@dataclass
class VoxelGridAndGoals:
    voxel_grid: VoxelGrid
    goals_out_of_grid: list[Goal]
    goals_in_negative_weight_grid: list[Goal]
    goals_in_zero_weight_grid: list[Goal]
    goals_in_positive_weight_grid: list[Goal]


@st.composite
def points_in_voxels(
    draw: st.DrawFn,
    voxel_size: float,
    voxels: pd.MultiIndex,
) -> list[npt.NDArray[np.floating]]:
    points = []
    for voxel in list(voxels):
        n_points_in_voxel = draw(
            st.integers(min_value=1, max_value=MAX_POINTS_PER_VOXEL)
        )
        for _ in range(n_points_in_voxel):
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
            points.append(point)
    return points


@st.composite
def voxel_grid_and_goals(
    draw: st.DrawFn,
    weights_strategy: Callable[
        [int], st.SearchStrategy[npt.NDArray[np.floating]]
    ] = default_attention_system_weights,
) -> VoxelGridAndGoals:
    occuppied_voxel_limits = [MIN_VOXEL_COORDINATE, MAX_VOXEL_COORDINATE]
    unoccuppied_voxel_limits = [MAX_VOXEL_COORDINATE + 1, MAX_VOXEL_COORDINATE + 10]

    occuppied_voxel_grid = draw(
        default_voxel_grid(
            voxels_strategy=unique_voxels(
                min_voxels=1,
                min_voxel_coord=occuppied_voxel_limits[0],
                max_voxel_coord=occuppied_voxel_limits[1],
            ),
            weights_strategy=weights_strategy,
        )
    )
    unoccuppied_voxel_grid = draw(
        default_voxel_grid(
            voxels_strategy=unique_voxels(
                min_voxels=1,
                min_voxel_coord=unoccuppied_voxel_limits[0],
                max_voxel_coord=unoccuppied_voxel_limits[1],
            ),
        )
    )
    voxel_size = occuppied_voxel_grid.voxel_size

    unoccuppied_df = unoccuppied_voxel_grid.to_pandas()
    points_out_of_grid = draw(
        points_in_voxels(voxel_size=voxel_size, voxels=unoccuppied_df.index)
    )

    occuppied_df = occuppied_voxel_grid.to_pandas()
    negative_voxels = occuppied_df[occuppied_df["weight"] < 0.0].index
    points_in_negative_weight_grid = draw(
        points_in_voxels(voxel_size=voxel_size, voxels=negative_voxels)
    )

    zero_voxels = occuppied_df[occuppied_df["weight"] == 0.0].index
    points_in_zero_weight_grid = draw(
        points_in_voxels(voxel_size=voxel_size, voxels=zero_voxels)
    )

    positive_voxels = occuppied_df[occuppied_df["weight"] > 0.0].index
    points_in_positive_weight_grid = draw(
        points_in_voxels(voxel_size=voxel_size, voxels=positive_voxels)
    )
    return VoxelGridAndGoals(
        voxel_grid=occuppied_voxel_grid,
        goals_out_of_grid=draw(goals_at(points_out_of_grid)),
        goals_in_negative_weight_grid=draw(goals_at(points_in_negative_weight_grid)),
        goals_in_zero_weight_grid=draw(goals_at(points_in_zero_weight_grid)),
        goals_in_positive_weight_grid=draw(goals_at(points_in_positive_weight_grid)),
    )


@st.composite
def default_voxel_grid(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = voxel_sizes,
    voxels_strategy: st.SearchStrategy[list[Voxel]] | None = None,
    weights_strategy: Callable[
        [int], st.SearchStrategy[npt.NDArray[np.floating]]
    ] = default_attention_system_weights,
) -> VoxelGrid:
    """Return a voxel grid with a set of weights."""
    if voxels_strategy is None:
        voxels_strategy = unique_voxels()

    voxel_size = draw(voxel_size_strategy)
    voxels = draw(voxels_strategy)
    weights = draw(weights_strategy(len(voxels)))
    return VoxelGrid(
        voxel_size=voxel_size,
        voxels=voxels,
        weights=weights,
    )
