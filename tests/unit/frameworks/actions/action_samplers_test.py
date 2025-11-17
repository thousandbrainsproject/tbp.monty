# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest

from numpy.random import Generator, default_rng
from typing_extensions import Protocol

from tbp.monty.frameworks.actions.action_samplers import (
    ActionSampler,
    ConstantSampler,
    UniformlyDistributedSampler,
)
from tbp.monty.frameworks.actions.actions import Action, QuaternionWXYZ, VectorXYZ
from tbp.monty.frameworks.agents import AgentID

RNG_SEED = 1337


class FakeActionOneActionSampler(Protocol):
    def sample_fake_action_one(self, agent_id: AgentID) -> FakeActionOne: ...


class FakeActionOneActuator(Protocol):
    def actuate_fake_action_one(self, action: FakeActionOne) -> None: ...


class FakeActionOne(Action):
    @classmethod
    def sample(
        cls, agent_id: AgentID, sampler: FakeActionOneActionSampler
    ) -> FakeActionOne:
        return sampler.sample_fake_action_one(agent_id)

    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)

    def act(self, _: FakeActionOneActuator) -> None:
        pass


class FakeActionTwoActionSampler(Protocol):
    def sample_fake_action_two(self, agent_id: AgentID) -> FakeActionTwo: ...


class FakeActionTwoActuator(Protocol):
    def actuate_fake_action_two(self, action: FakeActionTwo) -> None: ...


class FakeActionTwo(Action):
    @classmethod
    def sample(
        cls, agent_id: AgentID, sampler: FakeActionTwoActionSampler
    ) -> FakeActionTwo:
        return sampler.sample_fake_action_two(agent_id)

    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)

    def act(self, _: FakeActionTwoActuator) -> None:
        pass


class FakeActionThreeActionSampler(Protocol):
    def sample_fake_action_three(self, agent_id: AgentID) -> FakeActionThree: ...


class FakeActionThreeActuator(Protocol):
    def actuate_fake_action_three(self, action: FakeActionThree) -> None: ...


class FakeActionThree(Action):
    @classmethod
    def sample(
        cls, agent_id: AgentID, sampler: FakeActionThreeActionSampler
    ) -> FakeActionThree:
        return sampler.sample_fake_action_three(agent_id)

    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)

    def act(self, _: FakeActionThreeActuator) -> None:
        pass


class FakeSampler(ActionSampler):
    def __init__(
        self,
        rng: Generator | None = None,
        actions: list[type[Action]] | None = None,
    ):
        super().__init__(rng=rng, actions=actions)

    def sample_fake_action_one(self, agent_id: AgentID) -> FakeActionOne:
        return FakeActionOne(agent_id)

    def sample_fake_action_two(self, agent_id: AgentID) -> FakeActionTwo:
        return FakeActionTwo(agent_id)

    def sample_fake_action_three(self, agent_id: AgentID) -> FakeActionThree:
        return FakeActionThree(agent_id)


class ActionSamplerTest(unittest.TestCase):
    def test_sample_samples_random_action(self) -> None:
        sampler = FakeSampler(
            actions=[FakeActionOne, FakeActionTwo, FakeActionThree],
            rng=default_rng(RNG_SEED),
        )
        actions = []
        for _ in range(100):
            actions.append(sampler.sample(AgentID("agent")))
            # We are counting on the fact that 100 random samples
            # will not all be the same

        # From 100 samples, we expect at least 2 different actions
        self.assertFalse(all(a.name == actions[0].name for a in actions))
        for action in actions:
            sampled_method_name = f"sample_{action.name}"
            sampled_method = getattr(sampler, sampled_method_name)
            action_again = sampled_method(AgentID("agent"))
            self.assertEqual(action.__dict__, action_again.__dict__)


class ConstantSamplerTest(unittest.TestCase):
    """Test the ConstantSampler class.

    The general approach is to create multiple actions of the same type and
    compare their parameters to ensure that they are the same across actions.
    """

    def setUp(self) -> None:
        self.absolute_degrees = 45.0
        self.direction: VectorXYZ = (1.0, 2.0, 3.0)
        self.location: VectorXYZ = (4.0, 5.0, 6.0)
        self.rotation_degrees = 4.0
        self.rotation_quat: QuaternionWXYZ = (0.0, 0.0, 0.0, 1.0)
        self.translation_distance = 0.02
        self.sampler = ConstantSampler(
            absolute_degrees=self.absolute_degrees,
            direction=self.direction,
            location=self.location,
            rng=default_rng(RNG_SEED),
            rotation_quat=self.rotation_quat,
            rotation_degrees=self.rotation_degrees,
            translation_distance=self.translation_distance,
        )

    def test_samples_look_down_with_constant_params(self) -> None:
        action1 = self.sampler.sample_look_down("agent1")
        action2 = self.sampler.sample_look_down("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )
        self.assertTrue(action1.constraint_degrees == action2.constraint_degrees == 90)

    def test_samples_look_up_with_constant_params(self) -> None:
        action1 = self.sampler.sample_look_up("agent1")
        action2 = self.sampler.sample_look_up("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )
        self.assertTrue(action1.constraint_degrees == action2.constraint_degrees == 90)

    def test_samples_move_forward_with_constant_params(self) -> None:
        action1 = self.sampler.sample_move_forward("agent1")
        action2 = self.sampler.sample_move_forward("agent2")
        self.assertTrue(
            action1.distance == action2.distance == self.translation_distance
        )

    def test_samples_move_tangentially_with_constant_params(self) -> None:
        action1 = self.sampler.sample_move_tangentially("agent1")
        action2 = self.sampler.sample_move_tangentially("agent2")
        self.assertTrue(
            action1.distance == action2.distance == self.translation_distance
        )
        self.assertEqual(action1.direction, self.direction)
        self.assertEqual(action2.direction, self.direction)

    def test_samples_orient_horizontal_with_constant_params(self) -> None:
        action1 = self.sampler.sample_orient_horizontal("agent1")
        action2 = self.sampler.sample_orient_horizontal("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )
        self.assertTrue(
            action1.left_distance == action2.left_distance == self.translation_distance
        )
        self.assertTrue(
            action1.forward_distance
            == action2.forward_distance
            == self.translation_distance
        )

    def test_samples_orient_vertical_with_constant_params(self) -> None:
        action1 = self.sampler.sample_orient_vertical("agent1")
        action2 = self.sampler.sample_orient_vertical("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )
        self.assertTrue(
            action1.down_distance == action2.down_distance == self.translation_distance
        )
        self.assertTrue(
            action1.forward_distance
            == action2.forward_distance
            == self.translation_distance
        )

    def test_samples_set_agent_pitch_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_agent_pitch("agent1")
        action2 = self.sampler.sample_set_agent_pitch("agent2")
        self.assertTrue(
            action1.pitch_degrees == action2.pitch_degrees == self.absolute_degrees
        )

    def test_samples_set_agent_pose_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_agent_pose("agent1")
        action2 = self.sampler.sample_set_agent_pose("agent2")
        self.assertEqual(action1.location, self.location)
        self.assertEqual(action1.rotation_quat, self.rotation_quat)
        self.assertEqual(action2.location, self.location)
        self.assertEqual(action2.rotation_quat, self.rotation_quat)

    def test_samples_set_sensor_pitch_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_pitch("agent1")
        action2 = self.sampler.sample_set_sensor_pitch("agent2")
        self.assertTrue(
            action1.pitch_degrees == action2.pitch_degrees == self.absolute_degrees
        )

    def test_samples_set_sensor_pose_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_pose("agent1")
        action2 = self.sampler.sample_set_sensor_pose("agent2")
        self.assertEqual(action1.location, self.location)
        self.assertEqual(action1.rotation_quat, self.rotation_quat)
        self.assertEqual(action2.location, self.location)
        self.assertEqual(action2.rotation_quat, self.rotation_quat)

    def test_samples_set_sensor_rotation_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_rotation("agent1")
        action2 = self.sampler.sample_set_sensor_rotation("agent2")
        self.assertEqual(action1.rotation_quat, self.rotation_quat)
        self.assertEqual(action2.rotation_quat, self.rotation_quat)

    def test_samples_set_yaw_with_constant_params(self) -> None:
        action1 = self.sampler.sample_set_yaw("agent1")
        action2 = self.sampler.sample_set_yaw("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.absolute_degrees
        )

    def test_samples_turn_left_with_constant_params(self) -> None:
        action1 = self.sampler.sample_turn_left("agent1")
        action2 = self.sampler.sample_turn_left("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )

    def test_samples_turn_right_with_constant_params(self) -> None:
        action1 = self.sampler.sample_turn_right("agent1")
        action2 = self.sampler.sample_turn_right("agent2")
        self.assertTrue(
            action1.rotation_degrees
            == action2.rotation_degrees
            == self.rotation_degrees
        )


class UniformlyDistributedSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.max_absolute_degrees = 360.0
        self.min_absolute_degrees = 180.0
        self.max_rotation_degrees = 45.0
        self.min_rotation_degrees = 15.0
        self.max_translation = 0.5
        self.min_translation = 0.3
        self.sampler = UniformlyDistributedSampler(
            max_absolute_degrees=self.max_absolute_degrees,
            min_absolute_degrees=self.min_absolute_degrees,
            rng=default_rng(RNG_SEED),
            max_rotation_degrees=self.max_rotation_degrees,
            min_rotation_degrees=self.min_rotation_degrees,
            max_translation=self.max_translation,
            min_translation=self.min_translation,
        )

    def test_samples_look_down_with_sampled_params(self) -> None:
        action1 = self.sampler.sample_look_down("agent1")
        action2 = self.sampler.sample_look_down("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )
            self.assertEqual(action.constraint_degrees, 90)

    def test_samples_look_up_with_sampled_params(self) -> None:
        action1 = self.sampler.sample_look_up("agent1")
        action2 = self.sampler.sample_look_up("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )
            self.assertEqual(action.constraint_degrees, 90)

    def test_samples_move_forward_with_sampled_params(self) -> None:
        action1 = self.sampler.sample_move_forward("agent1")
        action2 = self.sampler.sample_move_forward("agent2")
        self.assertNotAlmostEqual(action1.distance, action2.distance)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_translation <= action.distance <= self.max_translation
            )

    def test_samples_move_tangentially_with_sampled_params(self) -> None:
        action1 = self.sampler.sample_move_tangentially("agent1")
        action2 = self.sampler.sample_move_tangentially("agent2")
        self.assertNotAlmostEqual(action1.distance, action2.distance)
        for i in range(3):
            self.assertNotAlmostEqual(action1.direction[i], action2.direction[i])
        for action in [action1, action2]:
            self.assertTrue(
                self.min_translation <= action.distance <= self.max_translation
            )

    def test_samples_orient_horizontal_with_sample_params(self) -> None:
        action1 = self.sampler.sample_orient_horizontal("agent1")
        action2 = self.sampler.sample_orient_horizontal("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        self.assertNotAlmostEqual(action1.left_distance, action2.left_distance)
        self.assertNotAlmostEqual(action1.forward_distance, action2.forward_distance)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )
            self.assertTrue(
                self.min_translation <= action.left_distance <= self.max_translation
            )
            self.assertTrue(
                self.min_translation <= action.forward_distance <= self.max_translation
            )

    def test_samples_orient_vertical_with_sample_params(self) -> None:
        action1 = self.sampler.sample_orient_vertical("agent1")
        action2 = self.sampler.sample_orient_vertical("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        self.assertNotAlmostEqual(action1.down_distance, action2.down_distance)
        self.assertNotAlmostEqual(action1.forward_distance, action2.forward_distance)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )
            self.assertTrue(
                self.min_translation <= action.down_distance <= self.max_translation
            )
            self.assertTrue(
                self.min_translation <= action.forward_distance <= self.max_translation
            )

    def test_samples_set_agent_pitch_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_agent_pitch("agent1")
        action2 = self.sampler.sample_set_agent_pitch("agent2")
        self.assertNotAlmostEqual(action1.pitch_degrees, action2.pitch_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_absolute_degrees
                <= action.pitch_degrees
                <= self.max_absolute_degrees
            )

    def test_samples_set_agent_pose_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_agent_pose("agent1")
        action2 = self.sampler.sample_set_agent_pose("agent2")
        for i in range(3):
            self.assertNotAlmostEqual(action1.location[i], action2.location[i])
        for i in range(4):
            self.assertNotAlmostEqual(
                action1.rotation_quat[i], action2.rotation_quat[i]
            )

    def test_samples_set_sensor_pitch_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_pitch("agent1")
        action2 = self.sampler.sample_set_sensor_pitch("agent2")
        self.assertNotAlmostEqual(action1.pitch_degrees, action2.pitch_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_absolute_degrees
                <= action.pitch_degrees
                <= self.max_absolute_degrees
            )

    def test_samples_set_sensor_pose_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_pose("agent1")
        action2 = self.sampler.sample_set_sensor_pose("agent2")
        for i in range(3):
            self.assertNotAlmostEqual(action1.location[i], action2.location[i])
        for i in range(4):
            self.assertNotAlmostEqual(
                action1.rotation_quat[i], action2.rotation_quat[i]
            )

    def test_samples_set_sensor_rotation_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_sensor_rotation("agent1")
        action2 = self.sampler.sample_set_sensor_rotation("agent2")
        for i in range(4):
            self.assertNotAlmostEqual(
                action1.rotation_quat[i], action2.rotation_quat[i]
            )

    def test_samples_set_yaw_with_sample_params(self) -> None:
        action1 = self.sampler.sample_set_yaw("agent1")
        action2 = self.sampler.sample_set_yaw("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_absolute_degrees
                <= action.rotation_degrees
                <= self.max_absolute_degrees
            )

    def test_samples_turn_left_with_sample_params(self) -> None:
        action1 = self.sampler.sample_turn_left("agent1")
        action2 = self.sampler.sample_turn_left("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )

    def test_samples_turn_right_with_sample_params(self) -> None:
        action1 = self.sampler.sample_turn_right("agent1")
        action2 = self.sampler.sample_turn_right("agent2")
        self.assertNotAlmostEqual(action1.rotation_degrees, action2.rotation_degrees)
        for action in [action1, action2]:
            self.assertTrue(
                self.min_rotation_degrees
                <= action.rotation_degrees
                <= self.max_rotation_degrees
            )


if __name__ == "__main__":
    unittest.main()
