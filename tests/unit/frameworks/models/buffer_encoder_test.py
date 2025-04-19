# Copyright 2025 Thousand Brains Project
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

import numpy as np
import quaternion
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
from tbp.monty.frameworks.models.buffer import BufferEncoder
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
        quat = quaternion.quaternion(0, 1, 0, 0)
        self.assertEqual(
            json.loads(json.dumps(quat, cls=BufferEncoder)),
            quaternion.as_float_array(quat).tolist(),
        )

    def test_buffer_encoder_encodes_actions_by_default(self):
        actions = [
            LookDown(agent_id="test", rotation_degrees=47),
            LookUp(agent_id="test", rotation_degrees=77),
            TurnLeft(agent_id="test", rotation_degrees=90),
            TurnRight(agent_id="test", rotation_degrees=90),
        ]
        for action in actions:
            self.assertEqual(
                json.loads(json.dumps(action, cls=BufferEncoder)),
                json.loads(json.dumps(action, cls=ActionJSONEncoder)),
            )


if __name__ == "__main__":
    unittest.main()
