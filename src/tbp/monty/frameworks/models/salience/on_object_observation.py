# Copyright 2025-2026 Thousand Brains Project
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
import numpy.typing as npt

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation


@dataclass
class OnObjectObservation:
    center_location: npt.NDArray[np.float64] | None
    """3D location corresponding to the central pixel, if it is on an object."""
    locations: npt.NDArray[np.float64]
    """3D locations corresponding to on-object pixels."""
    salience: npt.NDArray[np.float32]
    """Salience values corresponding to on-object pixels."""
    on_object_map: npt.NDArray[np.bool_]
    """H x W boolean array indicating on-object pixels."""
    location_map: npt.NDArray[np.float64]
    """H x W x 3 array indicating the 3D location of each pixel."""


def on_object_observation(
    observation: SensorObservation,
    salience_map: np.ndarray,
) -> OnObjectObservation:
    """Convert all raw observation data into image format.

    This function reformats the arrays in a raw observations dictionary
    so that they're all indexable by image row and column indices. It also splits
    the semantic_3d array into 3D locations and an on-object/surface indicator array.

    Args:
        observation: A sensor observation.
        salience_map: A salience map.

    Returns:
        The grid/matrix formatted (unraveled) on-object salience and location data,
        along with the location corresponding to the central pixel.
    """
    rgba = observation["rgba"]
    grid_shape = rgba.shape[:2]
    semantic_3d = observation["semantic_3d"]
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
        on_object_map=on_object,
        location_map=locations,
    )
