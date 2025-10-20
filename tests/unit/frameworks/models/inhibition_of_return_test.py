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
        kernel_factory_class = DecayKernelFactory
        kernel_factory_args = {"tau_t": 10.0, "spatial_cutoff": 0.02, "w_t_min": 0.1}
        self.field = DecayField(
            kernel_factory_class=kernel_factory_class,
            kernel_factory_args=kernel_factory_args,
        )

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
        kernel_location_1 = np.array([1, 2, 3])
        kernel_location_2 = np.array([1.02, 2, 3])
        self.field.add(kernel_location_1)
        self.field.add(kernel_location_2)
        translation = np.array([0.001, 0.0, 0.0])
        for i in range(5):
            test_point_1 = kernel_location_1 + translation * i
            test_point_2 = kernel_location_2 - translation * i
            spatial_weights_1 = self.field.compute_weight(test_point_1)
            spatial_weights_2 = self.field.compute_weight(test_point_2)
            npt.assert_array_equal(spatial_weights_1, spatial_weights_2)

    def test_field_selects_max_from_overlapping_decaying_kernels(self) -> None:
        kernel_location_1 = np.array([1, 2, 3])
        kernel_location_2 = np.array([1.02, 2, 3])
        self.field.add(kernel_location_1)
        add_second_kernel_at_step = 10
        test_point = np.array([1.015, 2, 3])
        weights = []
        for step in range(30):
            if step == add_second_kernel_at_step:
                self.field.add(kernel_location_2)
            weights.append(self.field.compute_weight(test_point))
            self.field.step()

        weights_before_second_kernel = weights[:add_second_kernel_at_step]
        diffs_1 = np.ediff1d(weights_before_second_kernel)
        self.assertTrue(np.all(diffs_1 < 0))

        weights_after_second_kernel = weights[add_second_kernel_at_step:]
        diffs_2 = np.ediff1d(weights_after_second_kernel)
        self.assertTrue(np.all(diffs_2 < 0))

        w_before_second_kernel = weights_before_second_kernel[-1]
        w_after_second_kernel = weights_after_second_kernel[0]
        self.assertGreater(w_after_second_kernel, w_before_second_kernel)
