# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import inspect
import json
import unittest
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import quaternion as qt
import torch
from scipy.spatial.transform import Rotation

import tests.unit.frameworks.models.fakes.encoder_classes
from tbp.monty.frameworks.actions.actions import (
    ActionJSONEncoder,
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.buffer import BufferEncoder, FeatureAtLocationBuffer
from tests.unit.frameworks.models.fakes.encoder_classes import (
    FakeClass,
    FakeMixin,
    FakeSubclass1,
    FakeSubclass2,
    FakeSubclass3,
    FakeSubclass4,
)
from tests.unit.frameworks.models.fakes.encoders import (
    FakeJSONEncoder,
    fake_dict_encoder,
    fake_encoder,
    fake_list_encoder,
    fake_mixin_encoder,
)


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

    def test_get_all_features_on_object_pads_with_nans(self):
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

        features = self.buffer.get_all_features_on_object()

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

        # Row 2 for LM_step should be nans
        # This tests the padding is done with nan values.
        self.assertTrue(
            np.all(np.isnan(lm_pose_vectors[1])),
            "Expected nan padding for missing step 2, but got other values",
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


class PadToTargetLengthTest(unittest.TestCase):
    """Tests for FeatureAtLocationBuffer._pad_to_target_length method."""

    def setUp(self):
        self.buffer = FeatureAtLocationBuffer()
        # Add 3 steps to the buffer
        for i in range(3):
            state = create_mock_state(
                sender_id="SM_0",
                sender_type="SM",
                location=np.array([float(i), float(i), float(i)]),
                on_object=True,
            )
            self.buffer.append([state])

    def test_pads_shorter_array_to_buffer_length(self):
        # Create a 2x3 array (shorter than buffer length of 3)
        existing = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Pad to buffer length (should default to len(self) = 3)
        padded = self.buffer._pad_to_target_length(existing)

        # Should have 3 rows
        self.assertEqual(padded.shape[0], 3)
        self.assertEqual(padded.shape[1], 3)

        # First 2 rows should match existing data
        np.testing.assert_array_equal(padded[:2], existing[:2])

        # Last row should be all nans
        self.assertTrue(np.all(np.isnan(padded[2])))

    def test_returns_unchanged_when_already_at_target(self):
        # Create a 3x3 array (same length as buffer)
        existing = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        padded = self.buffer._pad_to_target_length(existing)
        np.testing.assert_array_equal(padded, existing)

    def test_returns_unchanged_when_longer_than_target(self):
        # Create a 5x3 array (longer than buffer length of 3)
        existing = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ]
        )

        padded = self.buffer._pad_to_target_length(existing)
        np.testing.assert_array_equal(padded, existing)

    def test_pads_with_explicit_target_length(self):
        # Create a 2x3 array
        existing = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Pad to explicit target length of 5
        padded = self.buffer._pad_to_target_length(existing, target_length=5)

        # Should have 5 rows (padded from 2 to 5)
        self.assertEqual(padded.shape[0], 5)

        # Columns should not have changed (i.e., 3)
        self.assertEqual(padded.shape[1], 3)

        # First 2 rows should match existing data
        np.testing.assert_array_equal(padded[:2], existing[:2])

        # Last 3 rows should be all nans
        self.assertTrue(np.all(np.isnan(padded[2:, :])))

    def test_pads_empty_array_with_explicit_new_val_len(self):
        # Create an empty array
        existing = np.empty((0, 0))

        # Pad with explicit new_val_len
        padded = self.buffer._pad_to_target_length(
            existing, target_length=3, new_val_len=5
        )

        # Rows padded to 3
        self.assertEqual(padded.shape[0], 3)

        # Columns extended to `new_val_len=5`
        self.assertEqual(padded.shape[1], 5)

        # All values should be nan
        self.assertTrue(np.all(np.isnan(padded)))

    def test_raises_error_for_empty_array_without_new_val_len(self):
        # Create an empty array
        existing = np.empty((0, 0))

        # Should raise ValueError when new_val_len is not provided
        with self.assertRaises(ValueError) as context:
            self.buffer._pad_to_target_length(existing)

        self.assertIn("Cannot infer width from empty array", str(context.exception))

    def test_raises_error_for_column_dimension_mismatch(self):
        # Create a 2x3 array
        existing = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Should raise ValueError when new_val_len conflicts with existing width
        with self.assertRaises(ValueError) as context:
            self.buffer._pad_to_target_length(existing, target_length=4, new_val_len=5)

        self.assertIn("Column dimension mismatch", str(context.exception))
        self.assertIn("has 3 columns", str(context.exception))
        self.assertIn("new_val_len=5", str(context.exception))


class BufferEncoderTest(unittest.TestCase):
    def setUp(self):
        """Ensure BufferEncoder does not contain any encoders created during tests."""
        for cls in inspect.getmembers(
            tests.unit.frameworks.models.fakes.encoder_classes,
            inspect.isclass,
        ):
            BufferEncoder.unregister(cls)

    def test_register_function_encoder(self):
        BufferEncoder.register(FakeClass, fake_encoder)
        fake = FakeClass(data=0)
        self.assertEqual(
            json.loads(json.dumps(fake, cls=BufferEncoder)),
            0,
        )

    def test_register_json_encoder_subclass_encoder(self):
        BufferEncoder.register(FakeClass, FakeJSONEncoder)
        fake = FakeClass(data=0)
        self.assertEqual(
            json.loads(json.dumps(fake, cls=BufferEncoder)),
            0,
        )

    def test_register_invalid_encoder_raises_value_error(self):
        with self.assertRaises(TypeError):
            BufferEncoder.register(FakeClass, None)

    def test_unregister_unregisters_encoder(self):
        BufferEncoder.register(FakeClass, fake_encoder)
        BufferEncoder.register(FakeSubclass1, fake_dict_encoder)
        fake_subclass_1 = FakeSubclass1(data=1)
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_1, cls=BufferEncoder)),
            {"data": 1},
        )
        BufferEncoder.unregister(FakeSubclass1)
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_1, cls=BufferEncoder)),
            1,
        )

    def test_encode_uses_method_resolution_order_encoder(self):
        BufferEncoder.register(FakeClass, fake_encoder)
        BufferEncoder.register(FakeMixin, fake_mixin_encoder)
        BufferEncoder.register(FakeSubclass1, fake_dict_encoder)
        BufferEncoder.register(FakeSubclass3, fake_list_encoder)
        fake = FakeClass(data=0)
        fake_subclass_1 = FakeSubclass1(data=1)
        fake_subclass_2 = FakeSubclass2(data=2)
        fake_subclass_3 = FakeSubclass3(data=3)
        fake_subclass_4 = FakeSubclass4(data=4)

        self.assertEqual(
            json.loads(json.dumps(fake, cls=BufferEncoder)),
            0,
        )
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_1, cls=BufferEncoder)),
            {"data": 1},
        )
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_2, cls=BufferEncoder)),
            {"data": 2},
        )
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_3, cls=BufferEncoder)),
            [3],
        )
        self.assertEqual(
            json.loads(json.dumps(fake_subclass_4, cls=BufferEncoder)),
            {"mixin": 4},
        )

    def test_buffer_encoder_encodes_numpy_ndarray_by_default(self):
        array = np.array([0, 1], dtype=int)
        self.assertEqual(
            json.loads(json.dumps(array, cls=BufferEncoder)),
            array.tolist(),
        )

    def test_buffer_encoder_encodes_numpy_ints_by_default(self):
        array = np.array([0, 1], dtype=int)
        for val in array:
            self.assertEqual(
                json.loads(json.dumps(val, cls=BufferEncoder)),
                val,
            )

    def test_buffer_encoder_encodes_numpy_bools_by_default(self):
        array = np.array([0, 1], dtype=int)
        for val in array.astype(bool):
            self.assertEqual(
                json.loads(json.dumps(val, cls=BufferEncoder)),
                val,
            )

    def test_buffer_encoder_encodes_scipy_rotation_by_default(self):
        rot = Rotation.from_euler("xyz", [30, 45, 60], degrees=True)
        self.assertEqual(
            json.loads(json.dumps(rot, cls=BufferEncoder)),
            rot.as_euler("xyz", degrees=True).tolist(),
        )

    def test_buffer_encoder_encodes_torch_tensors_by_default(self):
        tensor = torch.tensor([0, 1], dtype=torch.int32)
        self.assertEqual(
            json.loads(json.dumps(tensor, cls=BufferEncoder)),
            tensor.tolist(),
        )

    def test_buffer_encoder_encodes_quaternions_by_default(self):
        quat = qt.quaternion(0, 1, 0, 0)
        self.assertEqual(
            json.loads(json.dumps(quat, cls=BufferEncoder)),
            qt.as_float_array(quat).tolist(),
        )

    def test_buffer_encoder_encodes_actions_by_default(self):
        actions = [
            LookDown(agent_id=AgentID("test"), rotation_degrees=47),
            LookUp(agent_id=AgentID("test"), rotation_degrees=77),
            TurnLeft(agent_id=AgentID("test"), rotation_degrees=90),
            TurnRight(agent_id=AgentID("test"), rotation_degrees=90),
        ]
        for action in actions:
            self.assertEqual(
                json.loads(json.dumps(action, cls=BufferEncoder)),
                json.loads(json.dumps(action, cls=ActionJSONEncoder)),
            )


if __name__ == "__main__":
    unittest.main()
