# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnObjectObservation:
    center_location: np.ndarray | None
    locations: np.ndarray
    salience: np.ndarray


def on_object_observation(
    raw_observation: dict,
    salience_map: np.ndarray,
) -> OnObjectObservation:
    """Convert all raw observation data into image format.

    This function reformats the arrays in a raw observations dictionary
    so that they're all indexable by image row and column indices. It also splits
    the semantic_3d array into 3D locations and an on-object/surface indicator array.

    Args:
        raw_observation: A sensor's raw observations dictionary.
        salience_map: A salience map.

    Returns:
        The grid/matrix formatted (unraveled) on-object salience and location data,
        along with the location corresponding to the central pixel.
    """
    rgba = raw_observation["rgba"]
    grid_shape = rgba.shape[:2]
    semantic_3d = raw_observation["semantic_3d"]
    locations = semantic_3d[:, 0:3].reshape(grid_shape + (3,))
    on_object = semantic_3d[:, 3].reshape(grid_shape).astype(int) > 0

    center_is_on_object = on_object[rgba.shape[0] // 2, rgba.shape[1] // 2]
    if center_is_on_object:
        center_location = locations[rgba.shape[0] // 2, rgba.shape[1] // 2]
    else:
        center_location = None

    pix_rows, pix_cols = np.where(on_object)
    on_object_locations = locations[pix_rows, pix_cols]
    on_object_salience = salience_map[pix_rows, pix_cols]
    return OnObjectObservation(
        center_location=center_location,
        salience=on_object_salience,
        locations=on_object_locations,
    )
