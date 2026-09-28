# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.frameworks.models.buffer import BufferEncoder

Voxel = Tuple[int, int, int]


VOXEL_LEVELS = ("x", "y", "z")
"""Names of the row index levels. This is for a pandas DataFrame MultiIndex."""


def voxelize_points(
    voxel_size: float,
    points: npt.NDArray[np.floating],
) -> list[Voxel]:
    """Find the voxel containing each point.

    Voxels are half-open: a point on the face shared by two voxels belongs to
    the one with the larger coordinate, i.e. voxel v spans [v, v + 1) along
    each axis, in units of ``voxel_size``.

    Args:
        points: An (N, 3) array of points.
        voxel_size: Edge length of a voxel.

    Returns:
        The voxel of each point, in point order.

    Raises:
        ValueError: If ``points`` is not an (N, 3) array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be of shape (N, 3), got {points.shape}.")
    voxels = np.floor(points / voxel_size).astype(int)
    return list(map(tuple, voxels.tolist()))


def voxelize_and_bin_points(
    voxel_size: float,
    points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
) -> pd.DataFrame:
    """Tabulate each point's voxel alongside its weight.

    The table is the input to per-voxel aggregation: grouping it by
    ``voxel`` bins the points, with each group's index giving the indices
    of the points inside that voxel.

    Args:
        voxel_size: Edge length of a voxel.
        points: An (N, 3) array of points.
        weights: An (N,) array of weights, aligned with ``points``.

    Raises:
        ValueError: If ``weights`` is not an (N,) array.

    Returns:
        A frame with one row per point, holding the point's ``voxel`` and
        ``weight``.
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be of shape (N,), got {weights.shape}.")

    voxels = voxelize_points(voxel_size, points)
    return pd.DataFrame({"voxel": voxels, "weight": weights})


class VoxelGrid:
    """A sparse grid of per-voxel features over 3D space.

    Backed by a pandas DataFrame whose rows are the occupied voxels -- an
    (x, y, z) integer MultiIndex of voxel coordinates (lower corners) -- and
    a "weight" column.
    """

    _voxel_size: float
    _data: pd.DataFrame

    @classmethod
    def empty(cls, voxel_size: float) -> VoxelGrid:
        return VoxelGrid(voxel_size, [], np.array([], dtype=np.float64))

    @classmethod
    def from_pandas(cls, voxel_size: float, data: pd.DataFrame) -> VoxelGrid:
        """Create a voxel grid from a pandas DataFrame.

        Args:
            voxel_size: Edge length of a voxel.
            data: A DataFrame with a MultiIndex of (x, y, z) voxel coordinates and a
            "weight" column.

        Returns:
            A voxel grid.
        """
        validate_dataframe(data)
        grid = object.__new__(cls)
        grid._voxel_size = voxel_size
        grid._data = data
        return grid

    def __init__(
        self,
        voxel_size: float,
        voxels: list[Voxel],
        weights: npt.NDArray[np.floating],
    ) -> None:
        """Initialize the voxel grid.

        Args:
            voxel_size: Edge length of a voxel.
            voxels: The occupied voxels.
            weights: The weights of the occupied voxels.
        """
        self._voxel_size = voxel_size
        index = pd.MultiIndex.from_tuples(voxels, names=VOXEL_LEVELS)
        self._data = pd.DataFrame({"weight": weights}, index=index)
        validate_dataframe(self._data)

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel."""
        return self._voxel_size

    def to_pandas(self) -> pd.DataFrame:
        """Return the backing frame, not a copy."""
        return self._data

    def weights_at_points(
        self,
        points: npt.NDArray[np.floating],
        fill_value: float = np.nan,
    ) -> npt.NDArray[np.floating]:
        """Look up voxel's weight at each point.

        Args:
            points: A (N, 3) array of points.
            fill_value: The value reported for points whose voxel is not in
                the grid; defaults to NaN.

        Returns:
            An (N,) array of weights; ``fill_value`` is used for any points in
            unoccupied voxels.
        """
        voxel_index = voxelize_points(self._voxel_size, points)
        return (
            self._data["weight"].reindex(voxel_index, fill_value=fill_value).to_numpy()
        )

    def weights(self) -> npt.NDArray[np.floating]:
        """Return the weights of the occupied voxels."""
        return self._data["weight"].to_numpy()

    def __len__(self) -> int:
        """Return the number of occupied voxels."""
        return len(self._data)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the structure of the voxel grid DataFrame.

    Args:
        df: The DataFrame to validate.

    Raises:
        ValueError: If
          - ``df`` does not have a MultiIndex with level
            names ('x', 'y', 'z').
          - ``df``'s index is not unique.
          - ``df`` does not have a "weight" column.
          - ``df``'s "weight" column is not 1D.
    """
    if not isinstance(df.index, pd.MultiIndex) or not all(
        level in df.index.names for level in VOXEL_LEVELS
    ):
        raise ValueError(
            f"DataFrame must have a multi-index with level names {VOXEL_LEVELS}."
        )

    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique.")

    if "weight" not in df.columns:
        raise ValueError("DataFrame must have a 'weight' column.")

    weights = df["weight"].to_numpy()
    if not len(weights.shape) == 1:
        raise ValueError(f"weights must be of shape (N,), got {weights.shape}.")


def encode_voxel_grid(grid: VoxelGrid) -> dict:
    """Encode a voxel grid into a JSON-encodable dictionary.

    Args:
        grid: The grid to encode.

    Returns:
        A dictionary with the grid's ``voxel_size``, its occupied ``voxels``
        as a (V, 3) array, and their ``weight`` as a (V,) array.
    """
    df = grid.to_pandas()
    return {
        "voxel_size": grid.voxel_size,
        # As a (V, 3) array: the MultiIndex itself is not JSON-encodable.
        "voxels": df.index.to_frame(index=False).to_numpy(),
        "weight": df["weight"].to_numpy(),
    }


BufferEncoder.register(VoxelGrid, encode_voxel_grid)
