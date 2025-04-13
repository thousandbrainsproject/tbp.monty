# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.logging_utils import (
    compute_pose_error,  # Replace with your actual module
)


class TestComputePoseError(unittest.TestCase):
    """Unit tests for the compute_pose_error function."""

    def test_zero_error_single_rotation(self):
        """Test pose error is zero when rotations are identical."""
        rot = Rotation.from_euler("xyz", [45, 30, 60], degrees=True)
        error = compute_pose_error(rot, rot)
        self.assertAlmostEqual(error, 0.0, places=6)

    def test_known_small_angle_difference(self):
        """Test pose error matches the known angle between simple rotations."""
        rot1 = Rotation.from_euler("z", 0, degrees=True)
        rot2 = Rotation.from_euler("z", 10, degrees=True)
        error = compute_pose_error(rot1, rot2)
        self.assertAlmostEqual(error, np.deg2rad(10), places=6)

    def test_min_error_from_batch(self):
        """Test function returns the minimum error from a batch of predicted rotations.

        The target rotation is 12°, so the closest in the batch [0°, 10°, 20°] is 10°,
        with a 2° error.
        """
        rot_batch = Rotation.from_euler("z", [0, 10, 20], degrees=True)
        target = Rotation.from_euler("z", 12, degrees=True)
        error = compute_pose_error(rot_batch, target)
        self.assertAlmostEqual(error, np.deg2rad(2), places=6)

    def test_180_degree_rotation(self):
        """Test that the pose error for a 180-degree rotation is pi radians."""
        rot1 = Rotation.from_quat([0, 0, 1, 0])  # 180° rotation around z-axis
        rot2 = Rotation.identity()
        error = compute_pose_error(rot1, rot2)
        self.assertAlmostEqual(error, np.pi, places=6)

    def test_invalid_input_type(self):
        """Test that the function raises an error when given an invalid input type."""
        with self.assertRaises(TypeError):
            compute_pose_error("not a rotation", Rotation.identity())

        with self.assertRaises(TypeError):
            compute_pose_error(Rotation.identity(), "not a rotation")


if __name__ == '__main__':
    unittest.main()
