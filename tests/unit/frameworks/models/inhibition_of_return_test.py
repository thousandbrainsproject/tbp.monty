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

from tbp.monty.frameworks.models.inhibition_of_return import DecayKernel


class DecayKernelTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_kernel_spatial_weightdecays_with_distance(self) -> None:
        point = np.array([1, 2, 3])
        kernel = DecayKernel(location=point)
        translation = np.array([0.001, 0.001, 0.001])

        prev_spatial_weight = kernel(point)
        npt.assert_allclose(prev_spatial_weight, 1.0)
        for _ in range(10):
            new_point = point + translation
            new_spatial_weight = kernel(new_point)
            self.assertLessEqual(new_spatial_weight, prev_spatial_weight)
            prev_spatial_weight = new_spatial_weight
