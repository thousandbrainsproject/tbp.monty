# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

from tbp.monty.frameworks.models.salience.on_object_observation import (
    OnObjectObservation,
    on_object_observation,
)


class OnObjectObservationTest(unittest.TestCase):
    def create_raw_observation(self, central_region_on_object: bool = True) -> dict:
        image_shape = (64, 64)
        on_object_rows = slice(32-5, 32+5)
        on_object_cols = slice(32-5, 32+5)

        rgba = np.zeros(image_shape + (4,), dtype=float)
        rgba[on_object_rows, on_object_cols] = np.array([1.0, 0.0, 0.0, 1.0])

        on_object = np.zeros(image_shape, dtype=bool)
        if central_region_on_object:
            on_object[on_object_rows, on_object_cols] = True

        locations = np.zeros((image_shape[0], image_shape[1], 3))
        for row in range(image_shape[0]):
            for col in range(image_shape[1]):
                locations[row, col] = np.array([row, col, 1.0])

        semantic_3d = np.zeros([image_shape[0] * image_shape[1], 4], dtype=float)
        semantic_3d[:, 0:3] = locations.reshape(-1, 3)
        semantic_3d[:, 3] = on_object.reshape(-1)


    def test_center_is_on_object_returns_with_center_location(self) -> None:
        pass
