# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.attention_system import DefaultAttentionSystem
from tbp.monty.attention.voxel_grid import Voxel, VoxelGrid

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

voxel_sizes = st.floats(min_value=MIN_VOXEL_SIZE, max_value=MAX_VOXEL_SIZE)

# point_coordinates = st.floats(
#     min_value=MIN_POINT_COORDINATE,
#     max_value=MAX_POINT_COORDINATE,
#     allow_nan=False,
#     width=64,
# )

# points_1d = arrays(dtype=np.float64, shape=(3,), elements=point_coordinates)
# points_2d = arrays(
#     dtype=np.float64,
#     shape=st.tuples(st.integers(min_value=1, max_value=MAX_POINTS), st.just(3)),
#     elements=point_coordinates,
# )


# def float_features_values(length: int) -> st.SearchStrategy[np.ndarray]:
#     return arrays(
#         dtype=np.float64,
#         shape=(length,),
#         elements=st.floats(min_value=-1e-6, max_value=1e6, exclude_min=True),
#         fill=st.just(1.0),
#     )


def valid_default_attention_system_weights(
    length: int,
) -> st.SearchStrategy[npt.NDArray[np.floating]]:
    return arrays(
        dtype=np.float64,
        shape=(length,),
        elements=st.floats(
            min_value=DefaultAttentionSystem.MIN_ATTENTION_WEIGHT,
            max_value=DefaultAttentionSystem.MAX_ATTENTION_WEIGHT,
        ),
    )


@st.composite
def unique_voxels(draw: st.DrawFn, min_voxels: int = 0) -> list[Voxel]:
    """Draw a list of unique voxels.

    Returns:
        List of unique voxel coordinates.
    """
    min_voxel_coord = MIN_VOXEL_COORDINATE
    max_voxel_coord = MAX_VOXEL_COORDINATE
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


# TODO: delete if unused once attention tests are completed
# @st.composite
# def unique_coordinate_bound_voxels(
#     draw: st.DrawFn, voxel_size: float, min_voxels: int = 0
# ) -> list[Voxel]:
#     """Draw a list of unique voxels that lie within coordinate bounds.

#     Returns:
#         List of unique voxel coordinates.
#     """
#     min_voxel_coord = int(-MAX_POINT_COORDINATE / voxel_size)
#     max_voxel_coord = int(MAX_POINT_COORDINATE / voxel_size)
#     voxel_axis_length = max_voxel_coord - min_voxel_coord + 1

#     min_total_voxels = min_voxels
#     max_total_voxels = min(voxel_axis_length**3, MAX_VOXELS)
#     return draw(
#         st.lists(
#             st.tuples(
#                 st.integers(
#                     min_value=min_voxel_coord,
#                     max_value=max_voxel_coord,
#                 ),
#                 st.integers(
#                     min_value=min_voxel_coord,
#                     max_value=max_voxel_coord,
#                 ),
#                 st.integers(
#                     min_value=min_voxel_coord,
#                     max_value=max_voxel_coord,
#                 ),
#             ),
#             min_size=min_total_voxels,
#             max_size=max_total_voxels,
#             unique=True,
#         )
#     )


@st.composite
def default_voxel_grid(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = voxel_sizes,
    voxels_strategy: st.SearchStrategy[list[Voxel]] | None = None,
) -> VoxelGrid:
    """Constructs a voxel grid with a set of weights.

    Returns:
       Voxel grid.
    """
    voxels_strategy = voxels_strategy or unique_voxels()

    voxel_size = draw(voxel_size_strategy)
    voxels = draw(voxels_strategy)
    weights = draw(valid_default_attention_system_weights(len(voxels)))
    return VoxelGrid(
        voxel_size=voxel_size,
        voxels=voxels,
        weights=weights,
    )
