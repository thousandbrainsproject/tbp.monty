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
from functools import partial
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Goal
from tbp.monty.experiment.environment import (
    Interface,
    OneObjectPerEpisodeInterface,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
)
from tbp.monty.frameworks.environments.environment import SimulatedObjectEnvironment
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.simulators.mujoco import MuJoCoSimulator
from tbp.monty.simulators.mujoco.agents import DistantAgent

AGENT_ID = AgentID("agent_id_0")
VIEW_FINDER_SENSOR_ID = SensorID("view_finder")

CUBE_LOCATION = (0.0, 1.5, -0.1)
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

    env: ClassVar[SimulatedObjectEnvironment]
    env_interface: ClassVar[Interface]
    motor_policy: ClassVar[MotorPolicy]
    patch_res: ClassVar[Resolution2D]
    view_finder_res: ClassVar[Resolution2D]

    observations: Observations
    proprioceptive_state: ProprioceptiveState

    @classmethod
    def setUpClass(cls) -> None:
        cls.patch_res = Resolution2D(64, 64)
        cls.view_finder_res = Resolution2D(64, 64)
        agent_partials = [
            partial(
                DistantAgent,
                agent_id=AGENT_ID,
                position=(0.0, 1.5, 0.2),
                sensor_configs={
                    SensorID("patch"): SensorConfig(
                        resolution=cls.patch_res,
                        zoom=10.0,
                    ),
                    VIEW_FINDER_SENSOR_ID: SensorConfig(
                        resolution=cls.view_finder_res,
                        zoom=1.0,
                    ),
                },
            ),
        ]

        cls.env = MuJoCoSimulator(agents=agent_partials)

        transforms = [
            DepthTo3DLocations(
                AGENT_ID,
                sensor_ids=[SensorID("patch"), VIEW_FINDER_SENSOR_ID],
                resolutions=[
                    (cls.patch_res.height, cls.patch_res.width),
                    (cls.view_finder_res.height, cls.view_finder_res.width),
                ],
                zooms=[10.0, 1.0],
                world_coord=True,
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]

        cls.env_interface = OneObjectPerEpisodeInterface(
            # We aren't calling code that uses object_names or the
            # object_init_sampler, so just mock it out.
            object_names=[],
            object_init_sampler=MagicMock(),
            env=cls.env,
            transform=transforms,
            experiment_mode=ExperimentMode.EVAL,
            rng=np.random.default_rng(42),
            seed=42,
        )

        # Since we aren't going through the full experiment process, we don't call
        # pre_epoch on the environment interface we built. This means we never add
        # an object to the scene using the object_init_sampler, which is why we mocked
        # it. We need to manually add the object to the scene for the test.
        cls.env.add_object(name="cubeSolid", position=CUBE_LOCATION)

        cls.motor_policy = LookAtGoal(AGENT_ID, VIEW_FINDER_SENSOR_ID)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.observations, self.proprioceptive_state = self.env_interface.step([])

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
    def test_saccades_to_goal_location(self, x: float, y: float, z: float) -> None:
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
            ctx=MagicMock(),
            observations=self.observations,
            state=MotorSystemState(self.proprioceptive_state),
            percept=MagicMock(),
            goal=goal,
        )
        # We're storing both of these values not only for the assertions below, but
        # for the next iteration of the Hypothesis test.
        self.observations, self.proprioceptive_state = self.env_interface.step(
            policy_result.actions
        )

        semantic_3d = self.observations[AGENT_ID][VIEW_FINDER_SENSOR_ID]["semantic_3d"]
        new_shape = (self.view_finder_res.height, self.view_finder_res.width, 3)
        xyz = semantic_3d[:, :3].reshape(new_shape)
        central_xyz = xyz[
            self.view_finder_res.height // 2, self.view_finder_res.width // 2
        ]
        distance = np.linalg.norm(central_xyz - goal_location)
        self.assertLess(distance, tolerance)
