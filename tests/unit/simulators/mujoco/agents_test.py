# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from unittest import TestCase

import numpy as np

from tbp.monty.simulators.mujoco.agents import (
    AgentID,
    SensorID,
    SingleSensorAgent,
)
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

PRIMITIVE_OBJECT_TYPES = [
    "box",
    "capsule",
    "cylinder",
    "ellipsoid",
    "sphere",
]


class MuJoCoAgentsTest(TestCase):
    """Basic tests for MuJoCo agents in the simulator."""

    def test_create_agent(self):
        agents = [
            SingleSensorAgent(
                agent_id=AgentID("1"),
                sensor_id=SensorID("1"),
            )
        ]
        with MuJoCoSimulator(agents) as sim:
            for obj_name in PRIMITIVE_OBJECT_TYPES:
                sim.remove_all_objects()
                sim.add_object(obj_name, position=(0.0, 1.5, -5.0))
                obs = sim.get_observations()
                agent_obs = obs[agents[0].agent_id]
                sensor_obs = agent_obs[agents[0].sensor_id]
                semantic = sensor_obs["semantic"]
                actual = np.unique(semantic[semantic >= 0])
                assert actual == 0  # the first item added is always ID 0


def normalize_depth(depth):
    """Normalize depth data for rendering.

    Depth data is floating point depth in meters so we need normalize it to be
    in the range of a single byte to be able to render it to greyscale.

    Returns:
        Pixel data to be rendered
    """
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2 * depth[depth <= 1].mean()
    # Scale to [0, 255]
    return (255 * np.clip(depth, 0, 1)).astype(np.uint8)


def normalize_semantic(semantic):
    # Display the contents of the first channel, which contains object
    # IDs. The second channel, seg[:, :, 1], contains object types.
    geom_ids = semantic[:, :, 0]
    # Infinity is mapped to -1
    geom_ids = geom_ids.astype(np.float64) + 1
    # Scale to [0, 1]
    geom_ids = geom_ids / geom_ids.max()
    pixels = 255 * geom_ids
    return pixels.astype(np.uint8)
