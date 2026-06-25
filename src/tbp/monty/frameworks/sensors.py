# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from dataclasses import dataclass
from typing import NewType

__all__ = ["Resolution2D", "SensorConfig", "SensorID"]

from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ

SensorID = NewType("SensorID", str)
"""Unique identifier for a sensor."""


@dataclass
class Resolution2D:
    """Pixel resolution of a sensor."""

    width: int
    height: int


@dataclass
class SensorConfig:
    """A sensor configuration, mapping to our configs in Hydra."""

    resolution: Resolution2D
    position: VectorXYZ = ZERO_VECTOR
    rotation: QuaternionWXYZ = IDENTITY_QUATERNION
    zoom: float = 1.0
