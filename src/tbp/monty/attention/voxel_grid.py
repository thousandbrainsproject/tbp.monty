# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

Voxel = Tuple[int, int, int]

# from tbp.monty.frameworks.models.buffer import BufferEncoder

# # Edge length of a voxel, in meters, when none is specified.
# DEFAULT_VOXEL_SIZE = 0.005

# # Names of the row index levels: a voxel's integer grid coordinate.
# VOXEL_LEVELS = ("x", "y", "z")

# # The feature every grid carries: the attention weight of each voxel.
# WEIGHT_FEATURE = "weight"


def voxelize_and_bin_points(
    voxel_size: float,
    points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating]
) -> pd.DataFrame:
    """Tabulate each point's voxel alongside its features.

    The table is the input to per-voxel aggregation: grouping it by
    ``voxel`` bins the points, with each group's index giving the indices
    of the points inside that voxel.

    Args:
        voxel_size: Edge length of a voxel.
        points: An (N, 3) array of points; a single flat point is accepted.
        weights: An (N,) array of weights, aligned with ``points``.

    Raises:
        ValueError: If ``points`` is not an (N, 3) array.

    Returns:
        A frame with one row per point, indexed by ``point_ind`` (the
        point's position in ``points``), holding the point's ``voxel`` and
        ``weight``.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be of shape (N, 3), got {pts.shape}.")

    weights = np.asarray(weights)
    if weights.ndim != 1:
        raise ValueError(f"weights must be of shape (N,), got {weights.shape}.")

    voxels_ = np.floor(pts / voxel_size).astype(int)
    voxels = list(map(tuple, voxels_.tolist()))
    df = pd.DataFrame({"voxel": voxels, "weight": weights})
    df.index.name = "point_ind"
    return df


# class VoxelGrid:
#     """A sparse grid of per-voxel features over 3D space.

#     Backed by a pandas DataFrame whose rows are the occupied voxels -- an
#     (x, y, z) integer MultiIndex of voxel coordinates (lower corners) -- and
#     whose columns are the features. Every grid carries the ``weight``
#     feature; a grid built without data has no voxels but keeps a typed,
#     empty ``weight`` column so it merges, looks up, and encodes like any
#     other grid. The attention system's ``decay`` updates a grid's weights in
#     place; its ``merge`` and ``expire`` build new grids from old ones.

#     A grid built from one step's proposals may carry the ``inhibit_all``
#     signal (see ``AttentionRegion``): a request to the merge to inhibit
#     everything, which the merged result itself never carries.
#     """

#     def __init__(
#         self,
#         voxel_size: float,
#         data: pd.DataFrame | None = None,
#         inhibit_all: bool = False,
#     ):
#         """Initialize the voxel grid.

#         Args:
#             voxel_size: Edge length of a voxel, in meters.
#             data: The backing frame, indexed by voxel with one column per
#                 feature; an empty grid when None.
#             inhibit_all: Whether the grid carries the inhibit-all signal.
#         """
#         self._voxel_size = voxel_size
#         self._inhibit_all = inhibit_all
#         if data is None:
#             # The weight column must carry a numeric dtype: a bare empty
#             # column would be object dtype and poison later concats.
#             self._data = pd.DataFrame(
#                 {WEIGHT_FEATURE: pd.Series(dtype=float)},
#                 index=pd.MultiIndex.from_arrays([[], [], []], names=VOXEL_LEVELS),
#             )
#         else:
#             self._data = data

#     @property
#     def voxel_size(self) -> float:
#         """Edge length of a voxel, in meters."""
#         return self._voxel_size

#     @property
#     def inhibit_all(self) -> bool:
#         """Whether the grid asks the merge to inhibit everything."""
#         return self._inhibit_all

#     @property
#     def index(self) -> pd.MultiIndex:
#         """The occupied voxel coordinates, as pandas multi-index."""
#         return self._data.index

#     @property
#     def features(self) -> tuple[str]:
#         """The names of the features (columns) in the grid."""
#         return tuple(self._data.columns)

#     def copy(self) -> VoxelGrid:
#         """Return a (deep) copy of the voxel grid."""
#         return VoxelGrid(self._voxel_size, self._data.copy(), self._inhibit_all)

#     def to_pandas(self) -> pd.DataFrame:
#         """Return the backing frame, not a copy.

#         Returns:
#             The frame indexed by (x, y, z) voxel, one column per feature.
#         """
#         return self._data

#     def contains_points(
#         self, points: npt.NDArray[np.floating]
#     ) -> npt.NDArray[np.bool_]:
#         """Test which points fall within an occupied voxel.

#         Args:
#             points: A (N, 3) array of points; a single flat point is accepted.

#         Returns:
#             A (N,) boolean array, True where the point's voxel is in the grid.
#         """
#         voxels: list[Voxel] = voxelize_points(points, self._voxel_size)
#         voxel_index = pd.MultiIndex.from_tuples(voxels, names=VOXEL_LEVELS)
#         return voxel_index.isin(self._data.index)

#     def feature_at_points(
#         self,
#         feature: str,
#         points: npt.NDArray[np.floating],
#         fill_value: float | None = None,
#     ) -> npt.NDArray:
#         """Look up a feature's value at each point.

#         Args:
#             feature: The feature (column) to look up.
#             points: A (N, 3) array of points; a single flat point is accepted.
#             fill_value: The value reported for points whose voxel is not in
#                 the grid; NaN when None.

#         Returns:
#             A (N,) array of the feature's values, ``fill_value`` where the
#             point's voxel is unoccupied.
#         """
#         voxel_index = voxelize_points(points, self._voxel_size)
#         # An explicit fill_value of None would fill with Python None (object
#         # dtype); NaN keeps the result numeric.
#         fill = np.nan if fill_value is None else fill_value
#         return self._data[feature].reindex(voxel_index, fill_value=fill).to_numpy()

#     def __getitem__(self, feature: str) -> pd.Series:
#         """Get a feature's values, one per occupied voxel.

#         Args:
#             feature: The feature (column) to select.

#         Returns:
#             The feature as a Series indexed by voxel.
#         """
#         assert isinstance(feature, str)
#         return self._data[feature]

#     def __len__(self) -> int:
#         """Return the number of occupied voxels."""
#         return len(self._data)


# def encode_voxel_grid(grid: VoxelGrid) -> dict:
#     """Encode a voxel grid into a JSON-encodable dictionary.

#     Args:
#         grid: The grid to encode.

#     Returns:
#         The grid's voxel size, its inhibit-all signal, its occupied voxels as
#         a (V, 3) array, and one (V,) array per feature column, keyed by the
#         feature name.
#     """
#     return {
#         "voxel_size": grid.voxel_size,
#         "inhibit_all": grid.inhibit_all,
#         # As a (V, 3) array: the MultiIndex itself is not JSON-encodable.
#         "voxels": grid.index.to_frame(index=False).to_numpy(),
#         **{feature: grid[feature].to_numpy() for feature in grid.features},
#     }


# BufferEncoder.register(VoxelGrid, encode_voxel_grid)
