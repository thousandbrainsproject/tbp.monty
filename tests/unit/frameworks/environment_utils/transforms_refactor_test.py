# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.environment_utils.transform_handlers import (
    GaussianBlurRGB,
    GaussianSmoothing,
    TransformContext,
    TransformMiddleware,
    TransformPipeline,
)
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.sensors import SensorID

SENSOR_ID = SensorID("0")
MAX_KERNEL = 15
BLUR_KERNEL_SIZES = [n for n in range(MAX_KERNEL + 1) if n == 0 or n % 2 == 1]


@st.composite
def rgba_and_blur_params(draw):
    """Generate a random RGBA float32 image with valid blur parameters.

    Returns:
        Tuple of (rgba array, sigma, kernel_size).
    """
    height = draw(st.integers(1, 128))
    width = draw(st.integers(1, 128))
    rgba = draw(
        arrays(
            dtype=np.float32,
            shape=(height, width, 4),
            elements=st.floats(min_value=0.0, max_value=255.0, width=32),
        )
    )
    sigma = draw(st.floats(min_value=0.1, max_value=10.0))
    kernel_size = draw(st.sampled_from(BLUR_KERNEL_SIZES))
    return rgba, sigma, kernel_size


class GaussianSmoothingTest(unittest.TestCase):
    def test_create_kernel(self):
        kernel_ground_truth = np.array(
            [
                [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
                [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                [0.03832756, 0.05576627, 0.06319146, 0.05576627, 0.03832756],
                [0.03382395, 0.04921356, 0.05576627, 0.04921356, 0.03382395],
                [0.02324684, 0.03382395, 0.03832756, 0.03382395, 0.02324684],
            ]
        )
        gaussian_smoother = GaussianSmoothing(None, sigma=2, kernel_width=5)

        self.assertTrue(
            (gaussian_smoother._kernel.shape == kernel_ground_truth.shape),
            "Kernel shapes do not match.",
        )

        all_equal = np.allclose(
            gaussian_smoother._kernel, kernel_ground_truth, atol=1e-5
        )
        self.assertTrue(all_equal, "Kernel values do not match.")

    def test_replication_padding(self):
        img = np.array([[1, 2, 3], [4, 5, 6]])
        padded_img_ground_truth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                [4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                [4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                [4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
            ]
        )
        gaussian_smoother = GaussianSmoothing(None, sigma=5, kernel_width=7)
        padded_img = gaussian_smoother._get_padded_img(img, pad_type="edge")

        self.assertTrue(
            (padded_img.shape == padded_img_ground_truth.shape),
            "Padded image shapes do not match.",
        )

        all_equal = np.allclose(padded_img, padded_img_ground_truth, atol=1e-5)
        self.assertTrue(all_equal, "Padding values do not match.")

    def test_gaussian_smoothing_preserves_image(self):

        img = np.ones((64, 64))
        gaussian_smoother = GaussianSmoothing(None, sigma=15, kernel_width=15)
        padded_img = gaussian_smoother._get_padded_img(img, pad_type="empty")
        filtered_img = gaussian_smoother._conv2d(padded_img, kernel_renorm=True)

        self.assertTrue(
            (img.shape == filtered_img.shape), "Filtered image shapes do not match."
        )

        all_equal = np.allclose(img, filtered_img, atol=1e-5)
        self.assertTrue(all_equal, "Filtered pixel values do not match.")

    def test_gaussian_smoothing_on_non_uniform_image(self):
        img = np.array([[1, 3, 7, 4], [5, 5, 2, 6], [4, 9, 3, 1], [2, 8, 5, 7]])

        filtered_img_ground_truth = np.array(
            [
                [3.37321446, 3.8278025, 4.5165925, 4.80560057],
                [4.45152116, 4.37522804, 4.41742389, 3.83576027],
                [5.42629393, 4.91199635, 5.0033111, 3.95219812],
                [5.53056036, 5.29703499, 5.50863473, 4.12873359],
            ]
        )

        gaussian_smoother = GaussianSmoothing(None, sigma=2, kernel_width=3)
        padded_img = gaussian_smoother._get_padded_img(img, pad_type="empty")
        filtered_img = gaussian_smoother._conv2d(padded_img, kernel_renorm=True)

        self.assertTrue(
            (filtered_img.shape == filtered_img_ground_truth.shape),
            "Filtered image shapes do not match.",
        )

        all_equal = np.allclose(filtered_img, filtered_img_ground_truth, atol=1e-5)
        self.assertTrue(all_equal, "Filtered pixel values do not match.")


class GaussianBlurRGBHandlerTest(unittest.TestCase):
    def test_negative_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size must be non-negative"):
            GaussianBlurRGB(next_transform=Mock(), sensor_id=SENSOR_ID, kernel_size=-1)

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size must be odd or 0"):
            GaussianBlurRGB(next_transform=Mock(), sensor_id=SENSOR_ID, kernel_size=4)

    def test_non_positive_sigma_with_auto_kernel_raises(self):
        with pytest.raises(
            ValueError, match="sigma must be positive when kernel_size is 0"
        ):
            GaussianBlurRGB(
                next_transform=Mock(), sensor_id=SENSOR_ID, sigma=0, kernel_size=0
            )

    def test_missing_rgba_raises(self):
        blur = GaussianBlurRGB(
            next_transform=Mock(), sensor_id=SENSOR_ID, sigma=1.0, kernel_size=3
        )
        with pytest.raises(KeyError, match="no 'rgba' key"):
            blur.call(SensorObservation())

    def test_blur_preserves_alpha(self):
        rng = np.random.RandomState(0)
        rgba = rng.randint(0, 256, size=(32, 32, 4), dtype=np.uint8)
        alpha_before = rgba[:, :, 3].copy()

        blur = GaussianBlurRGB(
            next_transform=Mock(), sensor_id=SENSOR_ID, sigma=1.5, kernel_size=3
        )
        result = blur.call(SensorObservation({"rgba": rgba.copy()}))["rgba"]

        self.assertEqual(result.shape, rgba.shape)
        np.testing.assert_array_equal(result[:, :, 3], alpha_before)

    @given(
        height=st.integers(min_value=1, max_value=256),
        width=st.integers(min_value=1, max_value=256),
        fill_value=st.integers(min_value=0, max_value=255),
        sigma=st.floats(min_value=0.1, max_value=10.0),
        kernel_size=st.sampled_from(BLUR_KERNEL_SIZES),
    )
    def test_blur_solid_image_returns_identical(
        self, height, width, fill_value, sigma, kernel_size
    ):
        """Convolution with any normalized kernel preserves a constant signal."""
        rgba_img = np.full((height, width, 4), fill_value, dtype=np.uint8)
        blur = GaussianBlurRGB(
            next_transform=Mock(),
            sensor_id=SENSOR_ID,
            sigma=sigma,
            kernel_size=kernel_size,
        )

        result_img = blur.call(SensorObservation({"rgba": rgba_img.copy()}))["rgba"]

        self.assertEqual(result_img.shape, rgba_img.shape)
        np.testing.assert_array_equal(result_img, rgba_img)

    @given(params=rgba_and_blur_params())
    def test_blur_reduces_total_variation(self, params):
        """Gaussian blur is a low-pass filter, so total variation cannot increase."""
        rgba, sigma, kernel_size = params

        def total_variation(img):
            img = img.astype(np.float64)
            return np.sum(np.abs(np.diff(img, axis=0))) + np.sum(
                np.abs(np.diff(img, axis=1))
            )

        blur = GaussianBlurRGB(
            next_transform=Mock(),
            sensor_id=SENSOR_ID,
            sigma=sigma,
            kernel_size=kernel_size,
        )
        result_rgba = blur.call(SensorObservation({"rgba": rgba.copy()}))["rgba"]
        result_rgb = result_rgba[:, :, :3]

        input_tv = total_variation(rgba[:, :, :3])
        result_tv = total_variation(result_rgb)

        self.assertTrue(
            result_tv < input_tv or np.allclose(result_tv, input_tv, atol=1e-5)
        )

    def test_pipeline_integration(self):
        pipeline = TransformPipeline(
            [
                TransformMiddleware(
                    GaussianBlurRGB,
                    sensor_id=SENSOR_ID,
                    sigma=1.0,
                    kernel_size=3,
                )
            ]
        )

        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[4, 4, :3] = 255
        rgba[:, :, 3] = 123

        result = pipeline(
            TransformContext(rng=np.random.RandomState(0)),
            SensorObservation({"rgba": rgba.copy()}),
        )["rgba"]

        self.assertEqual(result.shape, rgba.shape)
        np.testing.assert_array_equal(result[:, :, 3], rgba[:, :, 3])


if __name__ == "__main__":
    unittest.main()
