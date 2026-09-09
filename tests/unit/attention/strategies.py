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
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, MIN_ATTENTION_WEIGHT

# Generated data: at most MAX_POINTS points; point coordinates and voxel
# coordinates within +/- their MAX; voxel sizes in [MIN, MAX]_VOXEL_SIZE.
MIN_POINT_COORDINATE = -10
MAX_POINT_COORDINATE = 10
MAX_POINTS = 100
MAX_POINTS_PER_VOXEL = 10
MIN_VOXEL_SIZE = 0.001
MAX_VOXEL_SIZE = 1.0

# Generated points stay this far (as a fraction of a voxel) from voxel faces,
# so float error cannot move them into a neighbouring voxel.
VOXEL_EDGE_TOLERANCE = 1e-6

voxel_sizes = st.floats(min_value=MIN_VOXEL_SIZE, max_value=MAX_VOXEL_SIZE)

point_coordinates = st.floats(
    min_value=MIN_POINT_COORDINATE,
    max_value=MAX_POINT_COORDINATE,
    allow_nan=False,
    width=64,
)

points_1d = arrays(dtype=np.float64, shape=(3,), elements=point_coordinates)
points_2d = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=1, max_value=MAX_POINTS), st.just(3)),
    elements=point_coordinates,
)


def float_features_values(length: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=(length,),
        elements=st.floats(min_value=-1e-6, max_value=1e6, exclude_min=True),
        fill=st.just(1.0),
    )


def valid_weights(length: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=(length,),
        elements=st.floats(
            min_value=MIN_ATTENTION_WEIGHT, max_value=MAX_ATTENTION_WEIGHT
        ),
    )
