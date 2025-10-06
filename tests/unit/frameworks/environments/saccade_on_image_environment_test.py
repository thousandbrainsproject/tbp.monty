# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from pathlib import Path

import numpy as np

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.environments.two_d_data import SaccadeOnImageEnvironment

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"


class TwoDMovementTest(unittest.TestCase):
    DATA_PATH = (
        Path(__file__).parent.parent.parent / "resources" / "dataloader_test_images"
    )

    def setUp(self):
        self.env = SaccadeOnImageEnvironment(
            patch_size=48, data_path=str(self.DATA_PATH) + "/"
        )
        self.env.reset()
        self.current_state = self.env.get_state()
        self.prev_loc = self.current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]

    def test_move_forward(self):
        action = MoveForward(agent_id=AGENT_ID, distance=1)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertLess(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent should not have moved",
        )

    def test_move_backward(self):
        action = MoveForward(agent_id=AGENT_ID, distance=-1)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertLess(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent should not have moved",
        )

    def test_look_up(self):
        action = LookUp(agent_id=AGENT_ID, rotation_degrees=10)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertGreater(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent did not move",
        )

    def test_look_down(self):
        action = LookDown(agent_id=AGENT_ID, rotation_degrees=10)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertGreater(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent did not move",
        )

    def test_turn_left(self):
        action = TurnLeft(agent_id=AGENT_ID, rotation_degrees=10)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertGreater(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent did not move",
        )

    def test_turn_right(self):
        action = TurnRight(agent_id=AGENT_ID, rotation_degrees=10)
        _ = self.env.step([action])
        current_state = self.env.get_state()
        current_loc = current_state[AGENT_ID]["sensors"][SENSOR_ID + ".depth"][
            "position"
        ]
        self.assertGreater(
            np.linalg.norm(self.prev_loc - current_loc),
            0.0001,
            "Agent did not move",
        )
