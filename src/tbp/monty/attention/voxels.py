# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.frameworks.models.buffer import BufferEncoder

Voxel = Tuple[int, int, int]  # hashable voxel coordinates.


def as_array_voxels(voxels: Iterable) -> npt.NDArray[np.int_]:
    array = np.asarray(voxels, dtype=int)
    array = array.reshape(-1, 3) if array.ndim == 1 else array
    assert array.ndim == 2 and array.shape[1] == 3
    return array


def as_tuple_voxels(voxels: Iterable) -> tuple[Voxel, ...]:
    # Coerce to builtin ints: numpy scalars are not JSON-serializable, and these
    # tuples end up in telemetry.
    array_voxels = as_array_voxels(voxels)  # This does validation for us.
    return tuple(tuple(int(c) for c in row) for row in array_voxels)


def voxelize_and_bin_points(
    points: npt.NDArray[np.floating],
    voxel_size: float,
) -> dict[Voxel, list[int]]:
    """Quantize/bin 3D locations into voxels.

    Args:
        points: An array containing (x, y, z) coordinates.
        voxel_size: Edge length of a voxel.

    Returns:
        covoxel_points: A dictionary from voxel (x, y, z) to the the indices of the
            points inside it. (indices index into ``points``.) The keys of this
            dictionary are all the occupied voxels.

    """
    # as_tuple_voxels validates the shape and coerces to builtin ints, which these
    # keys need since they become index labels and end up in telemetry.
    voxels = as_tuple_voxels(np.floor(points / voxel_size))
    covoxel_points: dict[Voxel, list[int]] = defaultdict(list)
    for point_ind, voxel in enumerate(voxels):
        covoxel_points[voxel].append(point_ind)

    return covoxel_points
