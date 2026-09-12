# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Habitat-backed tests for the JumpToGoal undo check.

Unlike the tests in `motor_policies_test.py`, these tests render primitive 3D
objects with habitat-sim and build the post-jump percept with the standard
distant-agent pipeline (depth to 3D locations, camera sensor module), so that
`Message.get_on_object()` is observed rather than mocked. The case of interest is a
hypothesis-testing jump that lands the agent inside the object's geometry.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from unittest.mock import Mock

import habitat_sim
import numpy as np
import quaternion as qt
from unittest_parametrize import ParametrizedTestCase, parametrize

from tbp.monty.cmp import Message
from tbp.monty.frameworks.actions.actions import (
    SetAgentPose,
    SetSensorRotation,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.models.abstract_monty_classes import (
    Observations,
)
from tbp.monty.frameworks.models.motor_policies import (
    JumpToGoal,
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_system_state import (
    MotorSystemState,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.habitat import (
    PRIMITIVE_OBJECT_TYPES,
    HabitatSim,
    SingleSensorAgent,
)

AGENT_ID = AgentID("agent_id_0")
SENSOR_ID = SensorID("patch")
RESOLUTION = (64, 64)
# The agent starts at (0, 1.5, 0) looking down the negative z axis, so an object
# at this position is centered in the initial view.
OBJECT_POSITION = (0.0, 1.5, -0.35)


class JumpToGoalHabitatTest(ParametrizedTestCase):
    def setUp(self) -> None:
        self.agent = SingleSensorAgent(
            agent_id=AGENT_ID,
            sensor_id=SENSOR_ID,
            resolution=RESOLUTION,
            semantic=True,
        )
        self.sensor_module = CameraSM(
            sensor_module_id=SENSOR_ID,
            features=["on_object", "object_coverage", "pose_vectors"],
        )
        self.missing_to_max_depth = MissingToMaxDepth(agent_id=AGENT_ID, max_depth=1.0)
        self.depth_to_3d = DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=[SENSOR_ID],
            resolutions=[RESOLUTION],
            world_coord=True,
            get_all_points=True,
            use_semantic_sensor=True,
        )

    def percept(self, sim: HabitatSim, observations: Observations) -> Message:
        """Run raw habitat observations through the distant-agent sensor pipeline.

        Args:
            sim: The simulator the observations came from (for proprioceptive state).
            observations: Raw observations as returned by the simulator.

        Returns:
            The percept produced by the camera sensor module.
        """
        observations = self.missing_to_max_depth.call(observations)
        observations = self.depth_to_3d.call(observations, state=sim.states)
        return self.sensor_module.step(
            Mock(rng=np.random.default_rng(0)),
            observations[AGENT_ID][SENSOR_ID],
        )

    @parametrize(
        "object_name",
        [(name,) for name in ("cubeSolid", "capsule3DSolid", "icosphereSolid")],
    )
    def test_undoes_jump_that_lands_inside_object(self, object_name: str) -> None:
        with HabitatSim(agents=[self.agent]) as sim:
            sim.add_object(object_name, position=OBJECT_POSITION)

            observations, _ = sim.reset()
            pre_jump_state = MotorSystemState(sim.states)
            pre_jump_percept = self.percept(sim, observations)
            # Sanity check: the object is in view from the initial pose.
            self.assertTrue(pre_jump_percept.get_on_object())
            self.assertEqual(
                pre_jump_percept._semantic_id,
                PRIMITIVE_OBJECT_TYPES[object_name],
            )

            # Teleport the agent to the object's center, i.e. inside its geometry.
            sim.initialize_agent(
                AGENT_ID,
                habitat_sim.AgentState(
                    position=np.array(OBJECT_POSITION, dtype=np.float32)
                ),
            )
            observations, _ = sim.step([])
            post_jump_state = MotorSystemState(sim.states)
            raw_depth = observations[AGENT_ID][SENSOR_ID]["depth"]
            # Habitat renders nothing from inside a closed mesh: the depth at the
            # image center is 0, which the former depth-based check would have
            # treated as a good view.
            self.assertEqual(raw_depth[RESOLUTION[0] // 2, RESOLUTION[1] // 2], 0.0)
            post_jump_percept = self.percept(sim, observations)

        self.assertFalse(post_jump_percept.get_on_object())

        policy = JumpToGoal(AGENT_ID, SENSOR_ID)
        goal = Mock(
            location=np.array(OBJECT_POSITION),
            morphological_features={"pose_vectors": np.eye(3)},
        )
        jump_result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=pre_jump_state,
            percept=pre_jump_percept,
            goal=goal,
        )
        assert isinstance(jump_result, MotorPolicyResult)
        self.assertEqual(jump_result.status, PolicyStatus.IN_PROGRESS)

        undo_result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=post_jump_state,
            percept=post_jump_percept,
            goal=None,
        )
        assert isinstance(undo_result, MotorPolicyResult)
        self.assertEqual(undo_result.status, PolicyStatus.READY)
        self.assertEqual(len(undo_result.actions), 2)
        set_agent_pose = undo_result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        set_sensor_rotation = undo_result.actions[1]
        assert isinstance(set_sensor_rotation, SetSensorRotation)

        agent_state = pre_jump_state[AGENT_ID]
        np.testing.assert_array_equal(set_agent_pose.location, agent_state.position)
        np.testing.assert_array_equal(
            qt.as_float_array(set_agent_pose.rotation_quat),
            qt.as_float_array(agent_state.rotation),
        )
        np.testing.assert_array_equal(
            qt.as_float_array(set_sensor_rotation.rotation_quat),
            qt.as_float_array(agent_state.sensors[SENSOR_ID].rotation),
        )
