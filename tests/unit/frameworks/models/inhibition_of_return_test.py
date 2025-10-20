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

    def test_kernel_spatial_weight_decays_within_spatial_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        kernel = DecayKernel(location=location, spatial_cutoff=0.02)
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(20)])
        spatial_weights = kernel(points)
        for i in range(1, len(points)):
            self.assertLess(spatial_weights[i], spatial_weights[i - 1])

    def test_kernel_spatial_weight_decays_to_zero_outside_spatial_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        kernel = DecayKernel(location=location, spatial_cutoff=0.02)
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(20, 100)])
        spatial_weights = kernel(points)
        npt.assert_allclose(spatial_weights, 0.0)

    def test_kernel_temporal_weight_decays_with_time(self) -> None:
        location = np.array([1, 2, 3])
        kernel = DecayKernel(location=location)
        weights = []
        for _ in range(20):
            weights.append(kernel(location))
            kernel.step()
        for i in range(1, len(weights)):
            self.assertLess(weights[i], weights[i - 1])
