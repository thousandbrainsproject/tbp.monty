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
from unittest.mock import MagicMock, call

import numpy as np
import numpy.testing as nptest

from tbp.monty.frameworks.models.salience.return_inhibitor import (
    DecayField,
    DecayKernel,
    DecayKernelFactory,
    ReturnInhibitor,
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
        nptest.assert_allclose(spatial_weights, 0.0)

    def test_kernel_temporal_weight_decays_with_time(self) -> None:
        location = np.array([1, 2, 3])
        kernel = DecayKernel(location=location)
        weights = []
        for _ in range(20):
            weights.append(kernel(location.reshape(1, 3)))
            kernel.step()
        for i in range(1, len(weights)):
            nptest.assert_array_less(weights[i], weights[i - 1])


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
        spatial_weights = self.field.compute_weights(points)
        diffs = np.ediff1d(spatial_weights)
        self.assertTrue(np.all(diffs < 0))

    def test_single_kernel_weight_decays_to_zero_outside_spatial_cutoff(
        self,
    ) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(20, 100)])
        spatial_weights = self.field.compute_weights(points)
        nptest.assert_allclose(spatial_weights, 0.0)

    def test_single_kernel_weight_decays_within_temporal_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        weights = []
        for _ in range(34):
            weights.append(self.field.compute_weights(location.reshape(1, 3)))
            self.field.step()
        diffs = np.ediff1d(weights)
        self.assertTrue(np.all(diffs < 0))

    def test_single_kernel_weight_is_zero_beyond_temporal_cutoff(self) -> None:
        location = np.array([1, 2, 3])
        self.field.add(location)
        weights = []
        for i in range(100):
            if i > 34:  # after temporal cutoff
                weights.append(self.field.compute_weights(location.reshape(1, 3)))
            self.field.step()
        nptest.assert_allclose(weights, 0.0)

    def test_colocated_kernels_not_additive(self) -> None:
        location = np.array([1, 2, 3])
        translation = np.array([0.001, 0.0, 0.0])
        points = np.array([location + translation * i for i in range(100)])
        self.field.add(location)
        spatial_weights_1 = self.field.compute_weights(points)
        for _ in range(100):
            self.field.step()
            self.field.add(location)
            spatial_weights_2 = self.field.compute_weights(points)
            nptest.assert_array_equal(spatial_weights_1, spatial_weights_2)

    def test_field_selects_max_from_overlapping_kernels(self) -> None:
        kernel_location_1 = np.array([1, 2, 3])
        kernel_location_2 = np.array([1.02, 2, 3])
        self.field.add(kernel_location_1)
        self.field.add(kernel_location_2)
        translation = np.array([0.001, 0.0, 0.0])
        for i in range(5):
            test_point_1 = kernel_location_1 + translation * i
            test_point_2 = kernel_location_2 - translation * i
            spatial_weights_1 = self.field.compute_weights(test_point_1.reshape(1, 3))
            spatial_weights_2 = self.field.compute_weights(test_point_2.reshape(1, 3))
            nptest.assert_array_equal(spatial_weights_1, spatial_weights_2)

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
            weights.append(self.field.compute_weights(test_point.reshape(1, 3)))
            self.field.step()

        weights_before_second_kernel = weights[:add_second_kernel_at_step]
        diffs_1 = np.ediff1d(weights_before_second_kernel)
        self.assertTrue(np.all(diffs_1 < 0))

        weights_after_second_kernel = weights[add_second_kernel_at_step:]
        diffs_2 = np.ediff1d(weights_after_second_kernel)
        self.assertTrue(np.all(diffs_2 < 0))

        w_before_second_kernel = weights_before_second_kernel[-1]
        w_after_second_kernel = weights_after_second_kernel[0]
        nptest.assert_array_less(w_before_second_kernel, w_after_second_kernel)

    def test_field_returns_empty_array_if_empty_query(self) -> None:
        kernel_location = np.array([1, 2, 3])
        self.field.add(kernel_location)
        query_locations = np.array([]).reshape(0, 3)
        weights = self.field.compute_weights(query_locations)
        nptest.assert_array_equal(weights, np.array([]))

    def test_reset_clears_kernels(self) -> None:
        kernel_location = np.array([1, 2, 3])
        self.field.add(kernel_location)
        weights = self.field.compute_weights(kernel_location.reshape(1, 3))
        nptest.assert_array_equal(weights, 1.0)
        self.field.reset()
        weights = self.field.compute_weights(kernel_location.reshape(1, 3))
        nptest.assert_array_equal(weights, 0.0)


class ReturnInhibitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.return_inhibitor = ReturnInhibitor(decay_field_class=MagicMock)

    def test_return_inhibitor_add_central_location_if_present_to_decay_field(
        self,
    ) -> None:
        central_location = np.array([1, 2, 3])
        query_locations = np.array([[4, 5, 6]])
        self.return_inhibitor(central_location, query_locations)
        self.return_inhibitor._decay_field.add.assert_called_once_with(  # type: ignore[attr-defined]
            central_location
        )

    def test_return_inhibitor_does_not_add_central_location_if_not_present_to_decay_field(  # noqa: E501
        self,
    ) -> None:
        central_location = None
        query_locations = np.array([[4, 5, 6]])
        self.return_inhibitor(central_location, query_locations)
        self.return_inhibitor._decay_field.add.assert_not_called()  # type: ignore[attr-defined]

    def test_return_inhibitor_computes_weights_before_stepping_decay_field(
        self,
    ) -> None:
        central_location = np.array([1, 2, 3])
        query_locations = np.array([[4, 5, 6]])
        expected_calls = [
            call.add(central_location),
            call.compute_weights(query_locations),
            call.step(),
        ]

        self.return_inhibitor(central_location, query_locations)

        self.assertEqual(
            self.return_inhibitor._decay_field.method_calls,  # type: ignore[attr-defined]
            expected_calls,
        )

    def test_return_inhibitor_returns_computed_weights(self) -> None:
        central_location = np.array([1, 2, 3])
        query_locations = np.array([[4, 5, 6]])
        compute_weights_result = np.array([0.5])
        self.return_inhibitor._decay_field.compute_weights.return_value = (  # type: ignore[attr-defined]
            compute_weights_result
        )
        weights = self.return_inhibitor(central_location, query_locations)
        self.assertEqual(id(weights), id(compute_weights_result))

    def test_reset_resets_decay_field(self) -> None:
        self.return_inhibitor.reset()
        self.return_inhibitor._decay_field.reset.assert_called_once()  # type: ignore[attr-defined]
