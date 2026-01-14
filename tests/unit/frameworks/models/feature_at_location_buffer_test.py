# Copyright 2026 Thousand Brains Project
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
import numpy.typing as npt

from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer


def create_mock_state(
    sender_id: str,
    sender_type: str,
    location: npt.NDArray[np.float64],
    on_object: bool,
    pose_vectors: npt.NDArray[np.float64] = None,
):
    """Create a mock State object for testing the buffer.

    Args:
        sender_id: Input channel identifier.
        sender_type: Type of sender ("SM" or "LM").
        location: 3D location array.
        on_object: Whether the observation is on the object.
        pose_vectors: Optional pose vectors (3x3 array). Defaults to identity.

    Returns:
        A mock State object compatible with FeatureAtLocationBuffer.append().
    """
    if pose_vectors is None:
        pose_vectors = np.eye(3)

    state = Mock()
    state.sender_id = sender_id
    state.sender_type = sender_type
    state.location = location
    state.morphological_features = {
        "pose_vectors": pose_vectors.flatten(),
        "pose_fully_defined": True,
    }
    state.non_morphological_features = {}
    # For these tests focused on location/feature padding, we skip displacements.
    # displacements are computed and set by the LM's _add_displacements() method
    # before calling buffer.append().
    state.displacement = {}
    state.get_on_object = Mock(return_value=on_object)

    return state


class FeatureAtLocationBufferPaddingTest(unittest.TestCase):
    """Tests for FeatureAtLocationBuffer focusing on padding and filtering behavior."""

    def setUp(self):
        """Create a fresh buffer for each test."""
        self.buffer = FeatureAtLocationBuffer()

    def test_get_all_features_on_object_pads_with_nans_not_zeros(self):
        """Test that features are padded with nans when channel array is shorter.

        When a channel doesn't send data at certain steps, the feature array needs
        to be padded. This test ensures that the features array is padded to the
        correct length and values (i.e., nan values). Padding with nan values is
        necessary for downstream code that uses np.isnan() to identify and filter
        missing entries.
        """
        # Step 1: Both channels send data
        state_sm = create_mock_state(
            sender_id="SM_0",
            sender_type="SM",
            location=np.array([1.0, 2.0, 3.0]),
            on_object=True,
        )
        state_lm = create_mock_state(
            sender_id="LM_0",
            sender_type="LM",
            location=np.array([1.0, 2.0, 3.0]),
            on_object=True,
        )
        self.buffer.append([state_sm, state_lm])

        # Step 2: Only SM sends data
        state_sm_2 = create_mock_state(
            sender_id="SM_0",
            sender_type="SM",
            location=np.array([4.0, 5.0, 6.0]),
            on_object=True,
        )
        self.buffer.append([state_sm_2])

        # Get all features on object
        features = self.buffer.get_all_features_on_object()

        # LM_0 features should be padded with nans for step 2
        sm_pose_vectors = features["SM_0"]["pose_vectors"]
        lm_pose_vectors = features["LM_0"]["pose_vectors"]

        # Both channels should have 2 rows (one for each on-object step)
        # This tests that features are padded.
        self.assertEqual(sm_pose_vectors.shape[0], 2)
        self.assertEqual(lm_pose_vectors.shape[0], 2)

        # First row for both channels should have valid data
        self.assertFalse(np.any(np.isnan(sm_pose_vectors[0])))
        self.assertFalse(np.any(np.isnan(lm_pose_vectors[0])))

        # Row 2 for SM_step should be valid values (e.g., identity pose)
        self.assertFalse(np.any(np.isnan(sm_pose_vectors[1])))

        # Row 2 for LM_step should be nans (not zeros)
        # This tests the padding is done with nan values.
        self.assertTrue(
            np.all(np.isnan(lm_pose_vectors[1])),
            "Expected nan padding for missing step 2, but got zeros or other values",
        )

    def test_get_all_locations_on_object_pads_and_filters_like_features(self):
        """Test that locations are padded and filtered consistently with features.

        When get_all_locations_on_object() is called without an input_channel argument,
        it should pad each channel's locations to the full buffer length using nans,
        then filter by global_on_object_ids. This ensures the returned locations match
        the shape of features returned by get_all_features_on_object().
        """
        # Step 1: Both channels send data
        state_sm_1 = create_mock_state(
            sender_id="SM_0",
            sender_type="SM",
            location=np.array([1.0, 2.0, 3.0]),
            on_object=True,
        )
        state_lm_1 = create_mock_state(
            sender_id="LM_0",
            sender_type="LM",
            location=np.array([1.1, 2.1, 3.1]),
            on_object=True,
        )
        self.buffer.append([state_sm_1, state_lm_1])

        # Step 2: Only SM sends data
        state_sm_2 = create_mock_state(
            sender_id="SM_0",
            sender_type="SM",
            location=np.array([4.0, 5.0, 6.0]),
            on_object=True,
        )
        self.buffer.append([state_sm_2])

        # Get locations and features for comparison
        locations = self.buffer.get_all_locations_on_object()
        features = self.buffer.get_all_features_on_object()

        # Both should have the same channels
        self.assertEqual(set(locations.keys()), set(features.keys()))

        # For each channel, locations should have same number of rows as features
        for channel in locations.keys():
            loc_rows = locations[channel].shape[0]
            # Use any feature to compare (e.g., pose_vectors)
            feat_rows = features[channel]["pose_vectors"].shape[0]
            self.assertEqual(
                loc_rows,
                feat_rows,
                f"Channel {channel}: locations has {loc_rows} rows but "
                f"features has {feat_rows} rows",
            )


if __name__ == "__main__":
    unittest.main()
