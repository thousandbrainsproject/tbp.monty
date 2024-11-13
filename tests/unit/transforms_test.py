# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.environment_utils.transforms import GaussianSmoothing


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
        gaussian_smoother = GaussianSmoothing(agent_id=0, sigma=2, kernel_width=5)

        self.assertTrue(
            (gaussian_smoother.kernel.shape == kernel_ground_truth.shape),
            "Kernel shapes do not match.",
        )

        all_equal = np.allclose(
            gaussian_smoother.kernel, kernel_ground_truth, atol=1e-5
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
        gaussian_smoother = GaussianSmoothing(agent_id=0, sigma=5, kernel_width=7)
        padded_img = gaussian_smoother.get_padded_img(img, pad_type="edge")

        self.assertTrue(
            (padded_img.shape == padded_img_ground_truth.shape),
            "Padded image shapes do not match.",
        )

        all_equal = np.allclose(padded_img, padded_img_ground_truth, atol=1e-5)
        self.assertTrue(all_equal, "Padding values do not match.")

    def test_gaussian_smoothing(self):
        # TEST CASE 1
        img = np.ones((64, 64))
        gaussian_smoother = GaussianSmoothing(agent_id=0, sigma=15, kernel_width=15)
        padded_img = gaussian_smoother.get_padded_img(img, pad_type="empty")
        filtered_img = gaussian_smoother.conv2d(padded_img, kernel_renorm=True)

        self.assertTrue(
            (img.shape == filtered_img.shape), "Filtered image shapes do not match."
        )

        all_equal = np.allclose(img, filtered_img, atol=1e-5)
        self.assertTrue(all_equal, "Filtered pixel values do not match.")

        # TEST CASE 2
        img = np.array([[1, 3, 7, 4], [5, 5, 2, 6], [4, 9, 3, 1], [2, 8, 5, 7]])

        filtered_img_ground_truth = np.array(
            [
                [3.37321446, 3.8278025, 4.5165925, 4.80560057],
                [4.45152116, 4.37522804, 4.41742389, 3.83576027],
                [5.42629393, 4.91199635, 5.0033111, 3.95219812],
                [5.53056036, 5.29703499, 5.50863473, 4.12873359],
            ]
        )

        gaussian_smoother = GaussianSmoothing(agent_id=0, sigma=2, kernel_width=3)
        padded_img = gaussian_smoother.get_padded_img(img, pad_type="empty")
        filtered_img = gaussian_smoother.conv2d(padded_img, kernel_renorm=True)

        self.assertTrue(
            (filtered_img.shape == filtered_img_ground_truth.shape),
            "Filtered image shapes do not match.",
        )

        all_equal = np.allclose(filtered_img, filtered_img_ground_truth, atol=1e-5)
        self.assertTrue(all_equal, "Filtered pixel values do not match.")


if __name__ == "__main__":
    unittest.main()
