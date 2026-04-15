# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined
from tbp.monty.frameworks.experiments.monty_experiment import ExperimentMode
from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.habitat.agents import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import HabitatEnvironment

AGENT_ID = AgentID("agent_id_0")
VIEW_FINDER_SENSOR_ID = SensorID("view_finder")

CUBE_LOCATION = [0.0, 1.5, -0.1]
CUBE_FACE_CENTER_X = 0.0
CUBE_FACE_CENTER_Y = 1.5
CUBE_FACE_CENTER_Z = 0.0
CUBE_WIDTH = 0.2  # meters
CUBE_EDGE_PADDING_LENGTH = 0.01  # meters


class LookAtGoalTest(unittest.TestCase):
    """Tests saccading to correct goal location.

    Initializes an experiment / episode with a cube that has a large face that
    sits on the X-Y plane. Then supplies the policy with goals that are on that plane.
    Enacts the actions returned by the policy, and verifies that the observation after
    a goal is attempted is very close to the goal. Note: we must supply goals one after
    another -- i.e., not just checking that we can go from having the agent/sensor at
    its starting orientation to a single goal orientation.

    Another todo is to check whether we can look at goals that are behind the agent. I
    am not 100% confident that the conversion to euler angles that happens within the
    policy (which is actually necessary) will work correctly in that case.
    """

    @classmethod
    def setUpClass(cls):
        cls.view_finder_shape = [64, 64]
        env_init_args = {
            "agents": {
                "agent_args": {
                    "agent_id": AGENT_ID,
                    "sensor_ids": [SensorID("patch"), VIEW_FINDER_SENSOR_ID],
                    "height": 0.0,
                    "position": [0.0, 1.5, 0.2],
                    "resolutions": [[64, 64], cls.view_finder_shape],
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
                    "position": CUBE_LOCATION,
                }
            ],
            "data_path": None,
            "scene_id": None,
            "seed": 42,
        }
        cls.env = HabitatEnvironment(**env_init_args)

        transforms = [
            MissingToMaxDepth(AGENT_ID, max_depth=1, threshold=0.0),
            DepthTo3DLocations(
                AGENT_ID,
                sensor_ids=[SensorID("patch"), VIEW_FINDER_SENSOR_ID],
                resolutions=[[64, 64], cls.view_finder_shape],
                zooms=[10.0, 1.0],
                world_coord=True,
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]
        object_init_sampler = Predefined(
            positions=[CUBE_LOCATION],
            rotations=[[0.0, 0.0, 0.0]],
        )
        object_names = ["cubeSolid"]

        cls.env_interface = EnvironmentInterfacePerObject(
            object_names=object_names,
            object_init_sampler=object_init_sampler,
            env=cls.env,
            transform=transforms,
            experiment_mode=ExperimentMode.EVAL,
            rng=np.random.default_rng(42),
            seed=42,
        )

        cls.motor_policy = LookAtGoal(AGENT_ID, VIEW_FINDER_SENSOR_ID)
        cls.observations, cls.proprioceptive_state = cls.env_interface.step([])

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    @given(
        x=st.floats(
            min_value=CUBE_FACE_CENTER_X - CUBE_WIDTH / 2 + CUBE_EDGE_PADDING_LENGTH,
            max_value=CUBE_FACE_CENTER_X + CUBE_WIDTH / 2 - CUBE_EDGE_PADDING_LENGTH,
        ),
        y=st.floats(
            min_value=CUBE_FACE_CENTER_Y - CUBE_WIDTH / 2 + CUBE_EDGE_PADDING_LENGTH,
            max_value=CUBE_FACE_CENTER_Y + CUBE_WIDTH / 2 - CUBE_EDGE_PADDING_LENGTH,
        ),
        z=st.just(CUBE_FACE_CENTER_Z),
    )
    def test_saccades_to_goal_location(self, x, y, z):
        tolerance = 0.01  # 1 cm tolerance (euclidean distance)
        goal_location = np.array([x, y, z])
        goal = Goal(
            location=goal_location,
            morphological_features=None,
            non_morphological_features=None,
            confidence=1.0,
            use_state=True,
            sender_id="view_finder",
            sender_type="SM",
            goal_tolerances=None,
            info=None,
        )
        policy_result = self.motor_policy(
            ctx=None,
            observations=self.observations,
            state=self.proprioceptive_state,
            percept=None,
            goal=goal,
        )
        self.observations, self.proprioceptive_state = self.env_interface.step(
            policy_result.actions
        )

        semantic_3d = self.observations[AGENT_ID][VIEW_FINDER_SENSOR_ID]["semantic_3d"]
        xyz = semantic_3d[:, :3].reshape(self.view_finder_shape + [3])
        central_xyz = xyz[
            self.view_finder_shape[0] // 2, self.view_finder_shape[1] // 2
        ]
        distance = np.linalg.norm(central_xyz - goal_location)
        self.assertLess(distance, tolerance)
