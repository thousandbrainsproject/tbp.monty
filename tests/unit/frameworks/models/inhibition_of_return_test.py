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

from tbp.monty.frameworks.models.inhibition_of_return import (
    DecayField,
    DecayKernel,
    DecayKernelFactory,
)


class DecayKernelTest(unittest.TestCase):

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


class DecayFieldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel_factory = DecayKernelFactory(
            tau_t=10.0, spatial_cutoff=0.02, w_t_min=0.1
        )
        self.field = DecayField(self.kernel_factory)

    def test_single_kernel_weight_decays_within_spatial_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(20)])
        spatial_weights = self.field.compute_weight(points)
        diffs = np.ediff1d(spatial_weights)
        self.assertTrue(np.all(diffs < 0))

    def test_single_kernel_weight_decays_to_zero_outside_spatial_cutoff(
        self,
    ) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(20, 100)])
        spatial_weights = self.field.compute_weight(points)
        npt.assert_allclose(spatial_weights, 0.0)

    def test_single_kernel_weight_decays_within_temporal_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        weights = []
        for _ in range(34):
            weights.append(self.field.compute_weight(location))
            self.field.step()
        diffs = np.ediff1d(weights)
        self.assertTrue(np.all(diffs < 0))

    def test_single_kernel_weight_is_zero_beyond_temporal_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        weights = []
        for i in range(100):
            if i > 34:  # after temporal cutoff
                weights.append(self.field.compute_weight(location))
            self.field.step()
        npt.assert_allclose(weights, 0.0)

    def test_colocated_kernels_not_additive(self) -> None:
        location = np.array([1, 2, 3])
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(100)])
        self.field.add(location)
        spatial_weights_1 = self.field.compute_weight(points)
        for _ in range(100):
            self.field.step()
            self.field.add(location)
            spatial_weights_2 = self.field.compute_weight(points)
            npt.assert_array_equal(spatial_weights_1, spatial_weights_2)

    def test_field_selects_max_from_overlapping_kernels(self) -> None:
        location_1 = np.array([1, 2, 3])
        location_2 = np.array([1.02, 2, 3])
        test_point = np.array([1.01, 2, 3])
