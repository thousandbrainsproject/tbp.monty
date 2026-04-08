# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import pytest

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined
from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal
from tbp.monty.frameworks.sensors import SensorID

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.experiments.monty_experiment import ExperimentMode
from tbp.monty.simulators.habitat.agents import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import HabitatEnvironment

"""
Test I want to write:

Initialize an experiment / episode with cube (or plane) that has a large face that
sits on the X-Y plane. Then supply the policy with goals that are on that plane.
Enact the actions returned by the policy, and verify that the observation after
a goal is attempted is very close to the goal. Note: we must supply goals one after
another -- i.e., not just checking that we can go from having the agent/sensor at
its starting orientation to a single goal orientation.

I believe this is sufficient for verifying that the policy math is working.

Another todo is to check whether we can look at goals that are behind the agent. I
am not 100% confident that the conversion to euler angles that happens within the
policy (which is actually necessary) will work correctly in that case.
"""

class MotorPolicyTest(unittest.TestCase):
    def setUp(self):
        self.view_finder_shape = [64, 64]
        env_init_args = {
            "agents": {
                "agent_args": {
                    "agent_id": AgentID("agent_id_0"),
                    "sensor_ids": [SensorID("patch"), SensorID("view_finder")],
                    "height": 0.0,
                    "position": [0.0, 1.5, 0.2],
                    "resolutions": [[64, 64], self.view_finder_shape],
                    "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "rotations": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    "semantics": [False, False],
                    "zooms": [10.0, 1.0],
                },
                "agent_type": MultiSensorAgent,
            },
            "objects": [
                {
                    "name": "cubeSolid",
                    "position": [0.0, 1.5, -0.1],
                }
            ],
            "data_path": None,
            "scene_id": None,
            "seed": 42,
        }
        env = HabitatEnvironment(**env_init_args)

        transforms = [
            MissingToMaxDepth(AgentID("agent_id_0"), max_depth=1, threshold=0.0),
            DepthTo3DLocations(
                AgentID("agent_id_0"),
                sensor_ids=[SensorID("patch"), SensorID("view_finder")],
                resolutions=[[64, 64], self.view_finder_shape],
                zooms=[10.0, 1.0],
                world_coord=True,
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]
        object_init_sampler = Predefined(
            positions=[[0.0, 1.5, -0.1]],
            rotations=[[0.0, 0.0, 0.0]],
        )
        object_names = ["cubeSolid"]

        self.env_interface = EnvironmentInterfacePerObject(
            object_names=object_names,
            object_init_sampler=object_init_sampler,
            env=env,
            transform=transforms,
            experiment_mode=ExperimentMode.EVAL,
            rng=np.random.default_rng(42),
            seed=42,
        )

        self.motor_policy = LookAtGoal(AgentID("agent_id_0"), SensorID("sensor_id_0"))

    def test_motor_policy(self):
        pass
        # Flow: give policy a goal (and mock other args), get actions.
        # Step actions with env. Check observations central pixel.
