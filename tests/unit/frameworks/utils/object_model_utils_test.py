# Copyright 2026 Thousand Brains Project
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

from tbp.monty.frameworks.utils.object_model_utils import pose_vector_mean


class PoseVectorMeanTest(unittest.TestCase):
    def test_averages_normals_pointing_in_same_direction(self) -> None:
        pose_vecs = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        pose_fully_defined = np.array([[1.0], [1.0]])
        pv_mean, use_cds = pose_vector_mean(pose_vecs, pose_fully_defined)
        np.testing.assert_allclose(pv_mean[:3], [0, 0, 1])
        self.assertTrue(use_cds)

    def test_cancelling_normals_fall_back_to_first_valid_normal(self) -> None:
        """Regression test for opposite normals binned into the same voxel.

        When the observations in a voxel contain surface normals pointing in
        exactly opposite directions (e.g. the two sides of a thin surface) and
        the reference curvature directions cannot disambiguate them, the mean
        normal cancels to zero. pose_vector_mean should fall back to the first
        valid normal instead of raising.
        """
        # Degenerate (parallel) curvature directions make the half-sphere test
        # unable to separate the two opposite normals, so both get averaged.
        pose_vecs = np.array(
            [
                [0, 0, 1, 1, 0, 0, 1, 0, 0],
                [0, 0, -1, 1, 0, 0, 1, 0, 0],
            ],
            dtype=float,
        )
        pose_fully_defined = np.array([[0.0], [0.0]])
        pv_mean, use_cds = pose_vector_mean(pose_vecs, pose_fully_defined)
        np.testing.assert_allclose(pv_mean[:3], [0, 0, 1])
        self.assertFalse(use_cds)


if __name__ == "__main__":
    unittest.main()
