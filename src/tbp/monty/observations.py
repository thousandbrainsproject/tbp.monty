# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import TypedDict

import numpy as np
import numpy.typing as npt

__all__ = [
    "SensorObservation",
]


class SensorObservation(TypedDict, total=False):
    """Observations from a sensor."""

    rgba: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.float64]  # TODO: Verify specific type
    semantic: npt.NDArray[np.int_]  # TODO: Verify specific type
    semantic_3d: npt.NDArray[np.int_]  # TODO: Verify specific type
    sensor_frame_data: npt.NDArray[np.int_]  # TODO: Verify specific type
    cam_to_world: npt.NDArray[np.float64]  # TODO: Verify specific type
    pixel_loc: npt.NDArray[np.float64]  # TODO: Verify specific type
    raw: npt.NDArray[np.uint8]
