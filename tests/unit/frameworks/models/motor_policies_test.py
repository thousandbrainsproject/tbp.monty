# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nptest

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    OrientVertical,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    PredefinedPolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.sensors import SensorID


class BasePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(42)
        self.agent_id = AgentID(f"agent_id_{self.rng.randint(0, 999_999_999)}")
        self.default_sensor_state = SensorState(
            position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        )
        self.agent_sensors = {
            SensorID(
                f"sensor_id_{self.rng.randint(0, 999_999_999)}"
            ): self.default_sensor_state,
        }
        self.default_agent_state = AgentState(
            sensors=self.agent_sensors,
            position=self.default_sensor_state.position,
            rotation=self.default_sensor_state.rotation,
        )

        self.policy = BasePolicy(
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
        )

    def test_get_agent_state_selects_state_matching_agent_id(self):
        expected_state = AgentState(
            sensors=self.agent_sensors,
            position=self.default_sensor_state.position,
            rotation=self.default_sensor_state.rotation,
        )
        state = MotorSystemState(
            {
                self.agent_id: expected_state,
                AgentID("different_agent_id"): AgentState(
                    sensors={}, position=(), rotation=()
                ),
            }
        )
        self.assertEqual(self.policy.get_agent_state(state), expected_state)

    def test_is_motor_only_step_returns_false_if_motor_only_step_is_not_in_agent_state(
        self,
    ):
        state = MotorSystemState(
            {
                self.agent_id: self.default_agent_state,
            }
        )
        self.assertFalse(self.policy.is_motor_only_step(state))

    def test_is_motor_only_step_returns_true_if_motor_only_step_is_true_in_agent_state(
        self,
    ):
        state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors=self.default_agent_state.sensors,
                    position=self.default_agent_state.position,
                    rotation=self.default_agent_state.rotation,
                    motor_only_step=True,
                ),
            }
        )
        self.assertTrue(self.policy.is_motor_only_step(state))

    def test_is_motor_only_step_returns_false_if_motor_only_step_is_false_in_agent_state(  # noqa: E501
        self,
    ):
        state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors=self.default_agent_state.sensors,
                    position=self.default_agent_state.position,
                    rotation=self.default_agent_state.rotation,
                    motor_only_step=False,
                ),
            }
        )
        self.assertFalse(self.policy.is_motor_only_step(state))


class SurfacePolicyCurvatureInformedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.policy = SurfacePolicyCurvatureInformed(
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            desired_object_distance=0.025,
        )
        self.location = np.array([1.0, 2.0, 3.0])
        self.tangent_norm = np.array([0, 1, 0])
        self.state = State(
            location=self.location,
            morphological_features={
                "pose_vectors": np.array(
                    [self.tangent_norm.tolist(), [1, 0, 0], [0, 0, -1]]
                ),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures_log": [0, 0.5],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def test_assign_to_processed_observations_appends_to_tangent_locs_and_tangent_norms_if_last_action_is_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.actions = [
            OrientVertical(
                agent_id=self.agent_id,
                rotation_degrees=90,
                down_distance=1,
                forward_distance=1,
            )
        ]

        self.policy.processed_observations = self.state

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(len(self.policy.tangent_norms), 1)
        nptest.assert_array_equal(self.policy.tangent_norms[0], self.tangent_norm)

    def test_assign_to_processed_observations_appends_none_to_tangent_norms_if_last_action_is_orient_vertical_but_no_pose_vectors_in_state(  # noqa: E501
        self,
    ):
        del self.state.morphological_features["pose_vectors"]
        self.policy.actions = [
            OrientVertical(
                agent_id=self.agent_id,
                rotation_degrees=90,
                down_distance=1,
                forward_distance=1,
            )
        ]

        self.policy.processed_observations = self.state

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(self.policy.tangent_norms, [None])

    def test_assign_to_processed_observations_does_not_append_to_tangent_locs_and_tangent_norms_if_last_action_is_not_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.actions = [LookUp(agent_id=self.agent_id, rotation_degrees=0)]

        self.policy.processed_observations = self.state

        self.assertEqual(self.policy.tangent_locs, [])
        self.assertEqual(self.policy.tangent_norms, [])


class PredefinedPolicyReadActionFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.actions_file = Path(__file__).parent / "motor_policies_test_actions.jsonl"

    def test_read_action_file(self) -> None:
        # For this test, we write our own actions to a temporary file instead of
        # loading a file on disk. It's a better guarantee that we're loading the
        # actions exactly as expected.
        expected = [
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
            LookDown(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=10.0),
            LookUp(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
        ]
        with tempfile.TemporaryDirectory() as data_path:
            actions_file = Path(data_path) / "actions.jsonl"
            actions_file.write_text(
                "\n".join(json.dumps(a, cls=ActionJSONEncoder) for a in expected) + "\n"
            )
            loaded = PredefinedPolicy.read_action_file(actions_file)
            self.assertEqual(len(loaded), len(expected))
            for loaded_action, expected_action in zip(loaded, expected):
                self.assertEqual(dict(loaded_action), dict(expected_action))

    def test_cycles_continuously(self) -> None:
        policy = PredefinedPolicy(
            agent_id=self.agent_id,
            file_name=self.actions_file,
        )
        cycle_length = len(policy.action_list)
        ctx = RuntimeContext(rng=np.random.RandomState(42))
        returned_actions: list[Action] = []
        for _ in range(2 * cycle_length):
            result = policy.dynamic_call(ctx)
            assert len(result.actions) == 1, "Expected one action"
            returned_actions.append(result.actions[0])
            policy.post_actions(result.actions)

        for i in range(cycle_length):
            first_occurrence = returned_actions[i]
            second_occurrence = returned_actions[i + cycle_length]
            self.assertEqual(first_occurrence, second_occurrence)


if __name__ == "__main__":
    unittest.main()
