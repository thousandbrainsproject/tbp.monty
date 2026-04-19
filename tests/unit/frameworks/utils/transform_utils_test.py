# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
    rotation_from_quat,
)
from tbp.monty.geometry import Rotation, to_scalar_first, to_scalar_last


class RotationAsQuatTest(unittest.TestCase):
    def setUp(self):
        seed_quats = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-0.9606598, 0.07526898, -0.24576914, 0.10518961],
                [0.17179435, 0.52662329, -0.58935803, -0.58805759],
                [-0.0969695, -0.27647282, -0.86246987, -0.41268078],
                [-0.10221075, 0.89104681, -0.15463931, 0.41433709],
                [-0.5451644, -0.7671974, -0.00654478, 0.33787734],
            ]
        )
        self.rotations = Rotation.from_quat(seed_quats)
        self.quats_scalar_first = self.rotations.as_quat()
        self.quats_scalar_last = to_scalar_last(self.quats_scalar_first)

    def test_scalar_first_single(self):
        result = rotation_as_quat(self.rotations[0], scalar_first=True)
        expected = self.quats_scalar_first[0]
        np.testing.assert_array_equal(result, expected)

    def test_scalar_first_multiple(self):
        result = rotation_as_quat(self.rotations, scalar_first=True)
        expected = self.quats_scalar_first
        np.testing.assert_array_equal(result, expected)

    def test_scalar_last_single(self):
        result = rotation_as_quat(self.rotations[0], scalar_first=False)
        expected = self.quats_scalar_last[0]
        np.testing.assert_array_equal(result, expected)

    def test_scalar_last_multiple(self):
        result = rotation_as_quat(self.rotations, scalar_first=False)
        expected = self.quats_scalar_last
        np.testing.assert_array_equal(result, expected)

    def test_default_is_scalar_first(self):
        result = rotation_as_quat(self.rotations[0])
        expected = self.quats_scalar_first[0]
        np.testing.assert_array_equal(result, expected)


class RotationFromQuatTest(unittest.TestCase):
    def setUp(self):
        self.quats_scalar_first = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-0.9606598, 0.07526898, -0.24576914, 0.10518961],
                [0.17179435, 0.52662329, -0.58935803, -0.58805759],
                [-0.0969695, -0.27647282, -0.86246987, -0.41268078],
                [-0.10221075, 0.89104681, -0.15463931, 0.41433709],
                [-0.5451644, -0.7671974, -0.00654478, 0.33787734],
            ]
        )
        self.quats_scalar_last = to_scalar_last(self.quats_scalar_first)
        self.rotations = Rotation.from_quat(self.quats_scalar_first)

    def test_scalar_first_single(self):
        result = rotation_from_quat(self.quats_scalar_first[0], scalar_first=True)
        expected = self.rotations[0]
        np.testing.assert_array_equal(result.as_quat(), expected.as_quat())

    def test_scalar_first_multiple(self):
        result = rotation_from_quat(self.quats_scalar_first, scalar_first=True)
        expected = self.rotations
        np.testing.assert_array_equal(result.as_quat(), expected.as_quat())

    def test_scalar_last_single(self):
        result = rotation_from_quat(self.quats_scalar_last[0], scalar_first=False)
        expected = self.rotations[0]
        np.testing.assert_array_equal(result.as_quat(), expected.as_quat())

    def test_scalar_last_multiple(self):
        result = rotation_from_quat(self.quats_scalar_last, scalar_first=False)
        expected = self.rotations
        np.testing.assert_array_equal(result.as_quat(), expected.as_quat())

    def test_default_is_scalar_first(self):
        result = rotation_from_quat(self.quats_scalar_first[0])
        expected = self.rotations[0]
        np.testing.assert_array_equal(result.as_quat(), expected.as_quat())
