# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import unittest

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)

AGENT_ID = "camera"
SENSOR_ID = "sensor_01"

TEST_OBS = {
    AGENT_ID: {
        SENSOR_ID: {
            "semantic": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 5, 5, 5, 5, 0, 0],
                    [0, 0, 5, 5, 5, 5, 0, 0],
                    [0, 0, 5, 5, 5, 5, 0, 0],
                    [0, 0, 5, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            "depth": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        }
    }
}

EXPECTED_SEMANTIC_XY = np.array(
    [
        [-0.0857142, 0.0857142],
        [-0.0285714, 0.0857142],
        [0.0285714, 0.0857142],
        [0.0857142, 0.0857142],
        [-0.0857142, 0.0285714],
        [-0.0285714, 0.0285714],
        [0.0285714, 0.0285714],
        [0.0857142, 0.0285714],
        [-0.0857142, -0.0285714],
        [-0.0285714, -0.0285714],
        [0.0285714, -0.0285714],
        [0.0857142, -0.0285714],
        [-0.0857142, -0.0857142],
        [-0.0285714, -0.0857142],
        [0.0285714, -0.0857142],
        [0.0857142, -0.0857142],
    ]
)


class HabitatTransformTest(unittest.TestCase):
    def test_max_depth_transform(self):
        """Test replacing 0 with user specified max_range."""
        max_depth = 20
        transform = MissingToMaxDepth(
            agent_id=AGENT_ID, max_depth=max_depth, threshold=0
        )

        # Make a copy since this transform modifies the observation in place
        observation_copy = copy.deepcopy(TEST_OBS)

        # Control: get the 0 indices and check they are all zero
        m = np.where(observation_copy[AGENT_ID][SENSOR_ID]["depth"] <= 0)
        self.assertEqual(np.sum(observation_copy[AGENT_ID][SENSOR_ID]["depth"][m]), 0)

        # Check that the same indices get set to max_depth and only max_depth
        transformed_obs = transform(observation_copy)
        unique_0_replacements = np.unique(
            transformed_obs[AGENT_ID][SENSOR_ID]["depth"][m]
        )
        self.assertEqual(len(unique_0_replacements), 1)
        self.assertEqual(unique_0_replacements[0], max_depth)

    def test_semantic_3d_local(self):
        resolution = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"].shape
        # Replace 0 depth with max depth
        md_transform = MissingToMaxDepth(agent_id=AGENT_ID, max_depth=100)
        md_obs = md_transform(TEST_OBS)
        # Test transform using local coordinates
        transform = DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=[SENSOR_ID],
            resolutions=[resolution],
            use_semantic_sensor=True,
        )
        obs = transform(md_obs)
        module_obs = obs[AGENT_ID][SENSOR_ID]
        depth_obs = module_obs["depth"]
        semantic_obs = module_obs["semantic"]
        semantic_3d_obs = module_obs["semantic_3d"]

        # make sure old observations stay unchanged
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"]
        np.testing.assert_array_equal(depth_obs, expected)
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["semantic"]
        np.testing.assert_array_equal(semantic_obs, expected)

        # Check semantic 3D shape, it should be (nnz(semantic), 4)
        nnz = np.nonzero(semantic_obs)
        expected = (nnz[0].shape[0], 4)
        self.assertTupleEqual(semantic_3d_obs.shape, expected)

        # Check semantic id
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["semantic"]
        expected = np.unique(expected)
        expected = expected[expected.nonzero()][0]
        actual = np.unique(semantic_3d_obs[:, 3])
        self.assertEqual(actual, expected)

        # Check Z values, should match -depth
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"]
        expected = np.unique(expected)
        expected = -expected[expected.nonzero()][0]
        actual = np.unique(semantic_3d_obs[:, 2])
        self.assertEqual(actual, expected)

        # Check X,Y values
        actual = semantic_3d_obs[:, 0:2]
        np.testing.assert_array_almost_equal(actual, EXPECTED_SEMANTIC_XY)

    def setup_test_data(
        self, agent_position, agent_rotation, sensor_position, sensor_rotation
    ):
        resolution = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"].shape
        md_transform = MissingToMaxDepth(agent_id=AGENT_ID, max_depth=100)
        md_obs = md_transform(TEST_OBS)

        mock_state = {
            AGENT_ID: {
                "position": agent_position,
                "rotation": agent_rotation,
                "sensors": {
                    f"{SENSOR_ID}.depth": {
                        "position": sensor_position,
                        "rotation": sensor_rotation,
                    }
                },
            }
        }

        transform = DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=[SENSOR_ID],
            resolutions=[resolution],
            world_coord=True,
            get_all_points=False,
            use_semantic_sensor=True,
        )

        obs = transform(md_obs, state=mock_state)
        transformed_sensor_obs = obs[AGENT_ID][SENSOR_ID]
        depth_obs = transformed_sensor_obs["depth"]
        semantic_obs = transformed_sensor_obs["semantic"]
        semantic_3d_obs = transformed_sensor_obs["semantic_3d"]

        return md_obs, depth_obs, semantic_obs, semantic_3d_obs

    def compute_expected_semantic_3d(
        self,
        md_obs,
        semantic_obs,
        agent_position,
        agent_rotation,
        sensor_position,
        sensor_rotation,
    ):
        expected_x = EXPECTED_SEMANTIC_XY[:, 0]
        expected_y = EXPECTED_SEMANTIC_XY[:, 1]
        depth_values = md_obs[AGENT_ID][SENSOR_ID]["depth"][semantic_obs.nonzero()]
        expected_z = -depth_values

        points_camera = np.vstack(
            (expected_x, expected_y, expected_z, np.ones_like(expected_x))
        )

        agent_rotation_matrix = qt.as_rotation_matrix(agent_rotation)
        rotation = agent_rotation * sensor_rotation
        rotation_matrix = qt.as_rotation_matrix(rotation)

        translation = agent_position + agent_rotation_matrix @ sensor_position

        world_camera = np.eye(4)
        world_camera[0:3, 0:3] = rotation_matrix
        world_camera[0:3, 3] = translation

        points_world = (world_camera @ points_camera).T

        expected_semantic_id = np.unique(semantic_obs[semantic_obs.nonzero()])[0]

        expected_semantic_3d = np.column_stack(
            (
                points_world[:, 0],
                points_world[:, 1],
                points_world[:, 2],
                np.full(points_world.shape[0], expected_semantic_id),
            )
        )

        return expected_semantic_3d

    def test_semantic_3d_global_translation(self):
        agent_position = np.array([0.0, 0.0, 0.0])
        agent_rotation = qt.quaternion(1.0, 0.0, 0.0, 0.0)
        # translate the sensor relative to the agent along all 3 axes
        sensor_position = np.array([1.0, 0.5, -0.3])
        sensor_rotation = qt.quaternion(1.0, 0.0, 0.0, 0.0)

        md_obs, depth_obs, semantic_obs, semantic_3d_obs = self.setup_test_data(
            agent_position, agent_rotation, sensor_position, sensor_rotation
        )

        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"]
        np.testing.assert_array_equal(depth_obs, expected)
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["semantic"]
        np.testing.assert_array_equal(semantic_obs, expected)

        on_object_obs = np.nonzero(semantic_obs)
        expected_num_obs = (on_object_obs[0].shape[0], 4)
        self.assertTupleEqual(semantic_3d_obs.shape, expected_num_obs)

        expected_semantic_3d = self.compute_expected_semantic_3d(
            md_obs,
            semantic_obs,
            agent_position,
            agent_rotation,
            sensor_position,
            sensor_rotation,
        )

        np.testing.assert_array_almost_equal(semantic_3d_obs, expected_semantic_3d)

    def test_semantic_3d_global_agent_rotation(self):
        agent_position = np.array([0.0, 0.0, 0.0])
        x, y, z, w = Rotation.from_euler("xyz", [30, 45, -10], degrees=True).as_quat()
        # quaternion package uses w, x, y, z convention
        agent_rotation = qt.quaternion(w, x, y, z)
        sensor_position = np.array([0.0, 0.0, 0.0])
        sensor_rotation = qt.quaternion(1.0, 0.0, 0.0, 0.0)

        md_obs, depth_obs, semantic_obs, semantic_3d_obs = self.setup_test_data(
            agent_position, agent_rotation, sensor_position, sensor_rotation
        )

        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"]
        np.testing.assert_array_equal(depth_obs, expected)
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["semantic"]
        np.testing.assert_array_equal(semantic_obs, expected)

        on_object_obs = np.nonzero(semantic_obs)
        expected_num_obs = (on_object_obs[0].shape[0], 4)
        self.assertTupleEqual(semantic_3d_obs.shape, expected_num_obs)

        expected_semantic_3d = self.compute_expected_semantic_3d(
            md_obs,
            semantic_obs,
            agent_position,
            agent_rotation,
            sensor_position,
            sensor_rotation,
        )

        np.testing.assert_array_almost_equal(semantic_3d_obs, expected_semantic_3d)

    def test_semantic_3d_global_sensor_rotation(self):
        agent_position = np.array([0.0, 0.0, 0.0])
        agent_rotation = qt.quaternion(1.0, 0.0, 0.0, 0.0)
        sensor_position = np.array([0.0, 0.0, 0.0])
        x, y, z, w = Rotation.from_euler("xyz", [30, 45, -10], degrees=True).as_quat()
        sensor_rotation = qt.quaternion(w, x, y, z)

        md_obs, depth_obs, semantic_obs, semantic_3d_obs = self.setup_test_data(
            agent_position, agent_rotation, sensor_position, sensor_rotation
        )

        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["depth"]
        np.testing.assert_array_equal(depth_obs, expected)
        expected = TEST_OBS[AGENT_ID][SENSOR_ID]["semantic"]
        np.testing.assert_array_equal(semantic_obs, expected)

        on_object_obs = np.nonzero(semantic_obs)
        expected_num_obs = (on_object_obs[0].shape[0], 4)
        self.assertTupleEqual(semantic_3d_obs.shape, expected_num_obs)

        expected_semantic_3d = self.compute_expected_semantic_3d(
            md_obs,
            semantic_obs,
            agent_position,
            agent_rotation,
            sensor_position,
            sensor_rotation,
        )

        np.testing.assert_array_almost_equal(semantic_3d_obs, expected_semantic_3d)


if __name__ == "__main__":
    unittest.main()
