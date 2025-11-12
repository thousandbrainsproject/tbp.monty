# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import pytest

from tbp.monty.frameworks.agents import AgentID

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import unittest

import magnum as mn
import numpy as np
from habitat_sim._ext.habitat_sim_bindings.geo import FRONT

from tbp.monty.frameworks.actions.actions import LookDown, LookUp
from tbp.monty.simulators.habitat import HabitatSim, SingleSensorAgent


class HabitatSimTest(unittest.TestCase):
    def test_look_up_down_fix(self):
        sensor_id = "sensor_id_0"
        # Initialize the environment
        rotation_degrees = 10
        camera = SingleSensorAgent(
            agent_id=AgentID("camera"),
            sensor_id=sensor_id,
            resolution=(64, 64),
            translation_step=0.25,
            rotation_step=rotation_degrees,
        )
        with HabitatSim(agents=[camera]) as sim:
            # Retrieve agent
            agent = sim.get_agent(AgentID("camera"))
            scene_node = agent._sensors[f"{sensor_id}.rgba"].object

            # Test initial conditions
            rotation = scene_node.rotation
            look_vector = rotation.transform_vector(FRONT)
            look_angle = mn.Rad(np.arctan2(look_vector[1], -look_vector[2]))
            self.assertEqual(mn.Deg(look_angle), mn.Deg(0))

            # Look up until it reaches the defined constraint
            action = LookUp(
                agent_id=AgentID("camera"), rotation_degrees=rotation_degrees
            )
            constraint = action.constraint_degrees

            for _ in range(int(constraint * 2 / rotation_degrees) + 1):
                sim.apply_actions([action])

            rotation = scene_node.rotation
            look_vector = rotation.transform_vector(FRONT)
            look_angle = mn.Rad(np.arctan2(look_vector[1], -look_vector[2]))
            self.assertEqual(mn.Deg(look_angle), mn.Deg(constraint))

            # Look down until it reaches the defined constraint
            action = LookDown(
                agent_id=AgentID("camera"), rotation_degrees=rotation_degrees
            )
            constraint = action.constraint_degrees

            for _ in range(int(constraint * 2 / rotation_degrees) + 1):
                sim.apply_actions([action])

            rotation = scene_node.rotation
            look_vector = rotation.transform_vector(FRONT)
            look_angle = mn.Rad(np.arctan2(look_vector[1], -look_vector[2]))
            self.assertEqual(mn.Deg(look_angle), mn.Deg(-constraint))
