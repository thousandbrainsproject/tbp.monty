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
from unittest.mock import patch

import numpy as np
import numpy.testing as nptest

from tbp.monty.cmp import Message
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
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    PredefinedPolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tests.unit.frameworks.models.fakes.cmp import FakeMessage


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
        self.percept = Message(
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

    def test_appends_to_tangent_locs_and_tangent_norms_if_last_action_is_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(len(self.policy.tangent_norms), 1)
        nptest.assert_array_equal(self.policy.tangent_norms[0], self.tangent_norm)

    def test_appends_none_to_tangent_norms_if_last_action_is_orient_vertical_but_no_pose_vectors_in_state(  # noqa: E501
        self,
    ):
        del self.percept.morphological_features["pose_vectors"]
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(self.policy.tangent_norms, [None])

    def test_does_not_append_to_tangent_locs_and_tangent_norms_if_last_action_is_not_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = LookUp(
            agent_id=self.agent_id, rotation_degrees=0
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

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
        observations = Observations()
        returned_actions: list[Action] = []
        for _ in range(2 * cycle_length):
            result = policy(ctx, observations, MotorSystemState(), FakeMessage(), None)
            assert len(result.actions) == 1, "Expected one action"
            returned_actions.append(result.actions[0])

        for i in range(cycle_length):
            first_occurrence = returned_actions[i]
            second_occurrence = returned_actions[i + cycle_length]
            self.assertEqual(first_occurrence, second_occurrence)
