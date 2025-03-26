# Copyright 2025 Thousand Brains Project
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
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.states import GoalState, encode_goal_state


class EncodeGoalStateTest(unittest.TestCase):
    def setUp(self):
        self.goal_state_dict = {
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
            "use_state": True,
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
        self.goal_state = GoalState(**self.goal_state_dict)

    def test_encode(self):
        dct = encode_goal_state(self.goal_state)
        self.assertDictEqual(dct, self.goal_state_dict)

    def test_json_serialization(self):
        self.assertDictEqual(
            json.loads(json.dumps(self.goal_state, cls=BufferEncoder)),
            json.loads(json.dumps(self.goal_state_dict, cls=BufferEncoder)),
        )


if __name__ == "__main__":
    unittest.main()
