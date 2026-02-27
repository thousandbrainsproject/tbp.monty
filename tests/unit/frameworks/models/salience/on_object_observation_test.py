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
import numpy.testing as nptest

from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)


class OnObjectObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.center_location_sentinel = np.array([-1, -1, -1])

    def create_data(
        self,
        central_pixel_on_object: bool = True,
        central_region_on_object: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        image_shape = (64, 64)
        on_object_rows = slice(32 - 5, 32 + 5)
        on_object_cols = slice(32 - 5, 32 + 5)

        rgba = np.zeros(image_shape + (4,), dtype=float)

        on_object = np.zeros(image_shape, dtype=bool)
        if central_region_on_object:
            on_object[on_object_rows, on_object_cols] = True
            if not central_pixel_on_object:
                on_object[32, 32] = False

        locations = np.zeros((image_shape[0], image_shape[1], 3))
        for row in range(image_shape[0]):
            for col in range(image_shape[1]):
                locations[row, col] = np.array([row, col, 1.0])
        locations[32, 32] = self.center_location_sentinel

        semantic_3d = np.zeros([image_shape[0] * image_shape[1], 4], dtype=float)
        semantic_3d[:, 0:3] = locations.reshape(-1, 3)
        semantic_3d[:, 3] = on_object.reshape(-1)

        salience_map = np.ones(image_shape)

        pix_rows, pix_cols = np.where(on_object)
        on_object_locations = locations[pix_rows, pix_cols]
        on_object_salience = salience_map[pix_rows, pix_cols]

        return {
            "rgba": rgba,
            "semantic_3d": semantic_3d,
        }, {
            "on_object_locations": on_object_locations,
            "on_object_salience": on_object_salience,
            "salience_map": salience_map,
        }

    def test_center_is_on_object_returns_with_center_location(self) -> None:
        raw_observation, data = self.create_data(central_pixel_on_object=True)

        on_object = on_object_observation(raw_observation, data["salience_map"])
        self.assertIsNotNone(on_object.center_location)
        nptest.assert_array_equal(
            on_object.center_location, self.center_location_sentinel
        )  # type: ignore[arg-type]

    def test_center_is_not_on_object_returns_none_center_location(self) -> None:
        raw_observation, data = self.create_data(central_pixel_on_object=False)

        on_object = on_object_observation(raw_observation, data["salience_map"])
        self.assertIsNone(on_object.center_location)

    def test_on_object_salience_and_locations_are_correct(self) -> None:
        raw_observation, data = self.create_data(central_pixel_on_object=True)

        on_object = on_object_observation(raw_observation, data["salience_map"])
        nptest.assert_array_equal(on_object.salience, data["on_object_salience"])  # type: ignore[arg-type]
        nptest.assert_array_equal(on_object.locations, data["on_object_locations"])  # type: ignore[arg-type]

    def test_on_object_salience_and_locations_are_empty_if_no_on_object(self) -> None:
        raw_observation, data = self.create_data(central_region_on_object=False)

        on_object = on_object_observation(raw_observation, data["salience_map"])
        nptest.assert_array_equal(on_object.locations, np.zeros((0, 3)))  # type: ignore[arg-type]
        nptest.assert_array_equal(on_object.salience, np.zeros((0,)))  # type: ignore[arg-type]
