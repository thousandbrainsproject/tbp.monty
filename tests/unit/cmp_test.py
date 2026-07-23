# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import json
import unittest

import numpy as np

from tbp.monty.cmp import Goal, Message, encode_goal, location_mean
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.geometry import Rotation


def _message(sender_type: str, location) -> Message:
    return Message(
        location=location,
        morphological_features={},
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id=f"{sender_type}_0",
        sender_type=sender_type,
        process_features_in_lm=False,
    )


class CMPMessageTest(unittest.TestCase):
    def test_is_from_sm(self):
        self.assertTrue(_message("SM", np.zeros(3)).is_from_sm())
        self.assertFalse(_message("LM", np.zeros(3)).is_from_sm())

    def test_location_mean_averages_locations(self):
        messages = [
            _message("SM", np.array([0.0, 0.0, 0.0])),
            _message("SM", np.array([2.0, 4.0, 6.0])),
        ]
        np.testing.assert_array_equal(
            location_mean(messages), np.array([1.0, 2.0, 3.0])
        )

    def test_location_mean_none_when_no_location(self):
        self.assertIsNone(location_mean([]))
        self.assertIsNone(location_mean([_message("SM", None)]))


class EncodeGoalTest(unittest.TestCase):
    def setUp(self):
        self.goal_dict = {
            "location": np.array([0, 1.5, 0]),
            "morphological_features": {
                "pose_vectors": np.array(
                    [
                        -np.ones(3),
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            "non_morphological_features": None,
            "confidence": 1.0,
            "pass_message": True,
            "process_features_in_lm": True,
            "sender_id": "LM_0",
            "sender_type": "GSG",
            "goal_tolerances": None,
            "info": {
                "proposed_surface_loc": np.array([0, 1.5, 0]),
                "hypothesis_to_test": {
                    "graph_id": "mug",
                    "location": np.array([0, 1.5, 0]),
                    "rotation": Rotation.from_matrix(np.eye(3)),
                    "scale": 1.0,
                    "evidence": 1.0,
                },
                "achieved": False,
                "matching_step_when_output_goal_set": None,
            },
        }
        self.goal = Goal(**self.goal_dict)

    def test_encode(self):
        dct = encode_goal(self.goal)
        self.assertDictEqual(dct, self.goal_dict)

    def test_json_serialization(self):
        self.assertDictEqual(
            json.loads(json.dumps(self.goal, cls=BufferEncoder)),
            json.loads(json.dumps(self.goal_dict, cls=BufferEncoder)),
        )
