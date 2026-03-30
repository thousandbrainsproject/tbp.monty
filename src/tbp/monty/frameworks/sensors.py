# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import NewType, Tuple, TypedDict

__all__ = ["Resolution", "SensorConfig", "SensorID"]

from tbp.monty.math import QuaternionWXYZ, VectorXYZ

SensorID = NewType("SensorID", str)
"""Unique identifier for a sensor."""


# TODO: should this live elsewhere?
Resolution = Tuple[int, int]
"""Pixel resolution of a sensor."""


class SensorConfig(TypedDict):
    """A sensor configuration, mapping to our configs in Hydra."""

    position: VectorXYZ
    rotation: QuaternionWXYZ
    resolution: Resolution
    zoom: float
